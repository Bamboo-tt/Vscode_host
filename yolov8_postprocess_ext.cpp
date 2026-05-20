#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <limits>

namespace py = pybind11;

struct Box {
  float x1, y1, x2, y2;
};

static inline float sigmoid(float x) {
  // 与 python 版一致做 clip（避免 exp 溢出）
  if (x < -50.f) x = -50.f;
  if (x >  50.f) x =  50.f;
  return 1.f / (1.f + std::exp(-x));
}

static inline bool is_prob_like(const float* p, size_t n) {
  // python: 允许 [-1e-3, 1+1e-3] 视为概率 :contentReference[oaicite:7]{index=7}
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (size_t i=0;i<n;i++){
    mn = std::min(mn, p[i]);
    mx = std::max(mx, p[i]);
  }
  return (mn >= -1e-3f) && (mx <= 1.f + 1e-3f);
}

static std::vector<int> nms_xyxy(const std::vector<Box>& boxes,
                                 const std::vector<float>& scores,
                                 float iou_thres) {
  const int N = (int)boxes.size();
  std::vector<int> idx(N);
  for (int i=0;i<N;i++) idx[i]=i;
  std::sort(idx.begin(), idx.end(), [&](int a,int b){ return scores[a]>scores[b]; });

  std::vector<float> areas(N);
  for(int i=0;i<N;i++){
    float w = std::max(0.f, boxes[i].x2 - boxes[i].x1);
    float h = std::max(0.f, boxes[i].y2 - boxes[i].y1);
    areas[i] = w*h;
  }

  std::vector<int> keep;
  keep.reserve(N);

  for (int _i=0; _i<(int)idx.size(); _i++){
    int i = idx[_i];
    bool ok = true;
    for (int k : keep){
      float xx1 = std::max(boxes[i].x1, boxes[k].x1);
      float yy1 = std::max(boxes[i].y1, boxes[k].y1);
      float xx2 = std::min(boxes[i].x2, boxes[k].x2);
      float yy2 = std::min(boxes[i].y2, boxes[k].y2);
      float w = std::max(0.f, xx2-xx1);
      float h = std::max(0.f, yy2-yy1);
      float inter = w*h;
      float iou = inter / (areas[i] + areas[k] - inter + 1e-16f);
      if (iou > iou_thres) { ok=false; break; }
    }
    if (ok) keep.push_back(i);
  }
  return keep;
}

// 将任意 3D/4D 输出尽量视为 NCHW（简化版：假设 RKNN 输出多为 NCHW 或 NHWC）
// 你 Python 里做了更强的 _to_nchw 判断 :contentReference[oaicite:8]{index=8}。
// 这里要求：reg 必须能识别到 C=64，cls 必须能识别到 C=num_classes。
struct TensorView {
  // 仅支持 float32 / uint8 输入（RKNN 输出一般 float32）
  const float* fptr = nullptr;
  std::vector<ssize_t> shape; // N,C,H,W
  bool valid = false;
};

static TensorView to_nchw_f32(py::array arr, int num_classes){
  TensorView tv;
  py::buffer_info bi = arr.request();
  if (bi.ndim != 3 && bi.ndim != 4) return tv;

  // 强制转 float32（pybind 不做隐式转换）
  if (bi.format != py::format_descriptor<float>::format()) {
    // 让 python 侧先 .astype(np.float32)
    return tv;
  }
  tv.fptr = (const float*)bi.ptr;

  auto cand_c = std::vector<int>{1, 64, num_classes};

  if (bi.ndim == 4){
    ssize_t n=bi.shape[0], a=bi.shape[1], b=bi.shape[2], c=bi.shape[3];

    // NHWC -> NCHW：最后一维像通道
    auto is_cand = [&](ssize_t v){
      for(int cc: cand_c) if ((ssize_t)cc==v) return true;
      return false;
    };
    if (is_cand(c) && !is_cand(a)) {
      // 记录为 NCHW，但数据仍是 NHWC；这里不做转置，要求 RKNN 输出为 NCHW 更稳
      // 若你确实遇到 NHWC，需要补一个转置分支（会多拷贝）。
      return tv;
    }
    // NCHW
    tv.shape = {n, a, b, c};
    tv.valid = true;
    return tv;
  }

  // 3D：CHW
  tv.shape = {1, bi.shape[0], bi.shape[1], bi.shape[2]};
  tv.valid = true;
  return tv;
}

// softmax for reg vec length=16
static inline void softmax16(const float* x, float* out){
  float mx = x[0];
  for(int i=1;i<16;i++) mx = std::max(mx, x[i]);
  float sum=0.f;
  for(int i=0;i<16;i++){ out[i]=std::exp(x[i]-mx); sum += out[i]; }
  float inv = 1.f / (sum + 1e-12f);
  for(int i=0;i<16;i++) out[i] *= inv;
}

py::tuple yolov8_rknn_post_process_cpp(
    py::list outputs,
    int input_w,
    int input_h,
    int num_classes,
    float conf_thres,
    float iou_thres,
    int topk,
    bool person_only,
    int person_id,
    bool class_aware_nms
){
  // ---- 1) 分组：按 (H,W) 找 reg/cls ----
  struct Branch { int h,w; py::array reg; py::array cls; };
  std::unordered_map<long long, Branch> mp;

  auto key_hw = [](int h,int w)->long long{ return ((long long)h<<32) | (unsigned)w; };

  for (auto o : outputs){
    py::array arr = py::cast<py::array>(o);
    // python 里先 np.array(o) :contentReference[oaicite:9]{index=9}
    // 这里要求 float32
    if (arr.request().format != py::format_descriptor<float>::format()) {
      arr = py::array_t<float>(arr); // 尝试转换（可能复制）
    }

    py::buffer_info bi = arr.request();
    // 只支持 4D NCHW（最常见），否则先不处理
    if (bi.ndim != 4) continue;
    int n=(int)bi.shape[0], c=(int)bi.shape[1], h=(int)bi.shape[2], w=(int)bi.shape[3];
    if (n != 1) continue;

    auto k = key_hw(h,w);
    if (!mp.count(k)) mp[k] = Branch{h,w, py::array(), py::array()};

    if (c == 64) mp[k].reg = arr;
    else if (c == num_classes) mp[k].cls = arr;
  }

  std::vector<Branch> branches;
  branches.reserve(mp.size());
  for (auto &kv : mp){
    if (kv.second.reg.size() && kv.second.cls.size())
      branches.push_back(kv.second);
  }
  // 按 feature map 大小降序（80x80 -> 40x40 -> 20x20） :contentReference[oaicite:10]{index=10}
  std::sort(branches.begin(), branches.end(), [](const Branch& a,const Branch& b){
    return a.h*a.w > b.h*b.w;
  });

  std::vector<Box> all_boxes;
  std::vector<float> all_scores;
  std::vector<int> all_classes;
  all_boxes.reserve(2048);
  all_scores.reserve(2048);
  all_classes.reserve(2048);

  const int reg_max = 16;
  float acc[16];
  for(int i=0;i<16;i++) acc[i]=(float)i;

  // ---- 2) 遍历各尺度，做 score 筛选 + topk + DFL decode ----
  for (const auto& br : branches){
    auto reg = br.reg;
    auto cls = br.cls;

    py::buffer_info rbi = reg.request();
    py::buffer_info cbi = cls.request();
    const float* regp = (const float*)rbi.ptr; // [1,64,H,W]
    const float* clsp = (const float*)cbi.ptr; // [1,C,H,W]
    int H = (int)rbi.shape[2];
    int W = (int)rbi.shape[3];

    float stride_x = (float)input_w / (float)W;
    float stride_y = (float)input_h / (float)H;

    // 计算 score_map + cls_id_map（按你的 python 逻辑）:contentReference[oaicite:11]{index=11}
    std::vector<float> score_flat; score_flat.resize((size_t)H*W);
    std::vector<int> clsid_flat;  clsid_flat.resize((size_t)H*W);

    if (person_only){
      int cid = std::max(0, std::min(num_classes-1, person_id));
      const float* p = clsp + cid*(size_t)H*W; // cls[0,cid,:,:]
      bool prob = is_prob_like(p, (size_t)H*W);
      for (int i=0;i<H*W;i++){
        float v = p[i];
        v = prob ? v : sigmoid(v);
        score_flat[i]=v;
        clsid_flat[i]=cid;
      }
    } else {
      // argmax/max over C
      // 逐像素遍历：score = max(sigmoid/logit), clsid = argmax
      // 若 cls 已经是概率，跳过 sigmoid（同 python _maybe_prob）:contentReference[oaicite:12]{index=12}
      // 这里用“全局范围”判断概率；更精确可逐通道判断
      bool prob = true;
      {
        // 快速抽样判断（避免全扫 C*H*W 两遍）
        int sample = std::min(1024, H*W);
        for (int i=0;i<sample;i++){
          float v = clsp[i];
          if (v < -1e-3f || v > 1.f + 1e-3f){ prob=false; break; }
        }
      }
      for(int i=0;i<H*W;i++){
        float best=-1.f; int bestc=0;
        for(int c=0;c<num_classes;c++){
          float v = clsp[(size_t)c*H*W + (size_t)i];
          v = prob ? v : sigmoid(v);
          if (v>best){ best=v; bestc=c; }
        }
        score_flat[i]=best;
        clsid_flat[i]=bestc;
      }
    }

    // keep0: score >= conf_thres
    std::vector<int> keep0;
    keep0.reserve(H*W/10);
    for(int i=0;i<H*W;i++){
      if (score_flat[i] >= conf_thres) keep0.push_back(i);
    }
    if (keep0.empty()) continue;

    int k = std::min(topk, (int)keep0.size());
    // 选 topk：partial sort
    std::nth_element(keep0.begin(), keep0.begin()+k, keep0.end(), [&](int a,int b){
      return score_flat[a] > score_flat[b];
    });
    keep0.resize(k);
    std::sort(keep0.begin(), keep0.end(), [&](int a,int b){
      return score_flat[a] > score_flat[b];
    });

    // DFL decode：对候选点取 reg[0,:,y,x] -> 4*(16)
    float sm[16];
    for(int idx : keep0){
      int y = idx / W;
      int x = idx % W;
      float s = score_flat[idx];
      int c = clsid_flat[idx];

      // reg layout: [1,64,H,W] contiguous
      // 对每个边 (l,t,r,b)，取 16 bins
      float dist[4] = {0,0,0,0};
      for(int side=0; side<4; side++){
        const float* v = regp + (size_t)(side*16)*H*W + (size_t)y*W + (size_t)x;
        // v 是第一个 bin 的位置；后续 bin stride 是 H*W
        float bins[16];
        for(int b=0;b<16;b++){
          bins[b] = v[(size_t)b*H*W];
        }
        softmax16(bins, sm);
        float d=0.f;
        for(int b=0;b<16;b++) d += sm[b]*acc[b];
        dist[side]=d;
      }

      float cx = (float)x + 0.5f;
      float cy = (float)y + 0.5f;

      Box bb;
      bb.x1 = (cx - dist[0]) * stride_x;
      bb.y1 = (cy - dist[1]) * stride_y;
      bb.x2 = (cx + dist[2]) * stride_x;
      bb.y2 = (cy + dist[3]) * stride_y;

      // clip
      bb.x1 = std::max(0.f, std::min((float)input_w, bb.x1));
      bb.x2 = std::max(0.f, std::min((float)input_w, bb.x2));
      bb.y1 = std::max(0.f, std::min((float)input_h, bb.y1));
      bb.y2 = std::max(0.f, std::min((float)input_h, bb.y2));

      all_boxes.push_back(bb);
      all_scores.push_back(s);
      all_classes.push_back(c);
    }
  }

  if (all_boxes.empty()){
    auto boxes = py::array_t<float>(py::array::ShapeContainer{0, 4});
    auto classes = py::array_t<int>(py::array::ShapeContainer{0});
    auto scores = py::array_t<float>(py::array::ShapeContainer{0});
    return py::make_tuple(boxes, classes, scores);
  }

  // ---- 3) 全局 topk（避免 NMS 太慢）:contentReference[oaicite:13]{index=13}
  if ((int)all_scores.size() > topk){
    std::vector<int> idx(all_scores.size());
    for(size_t i=0;i<idx.size();i++) idx[i]=(int)i;
    std::nth_element(idx.begin(), idx.begin()+topk, idx.end(), [&](int a,int b){
      return all_scores[a] > all_scores[b];
    });
    idx.resize(topk);
    std::sort(idx.begin(), idx.end(), [&](int a,int b){ return all_scores[a]>all_scores[b]; });

    std::vector<Box> b2; b2.reserve(idx.size());
    std::vector<float> s2; s2.reserve(idx.size());
    std::vector<int> c2; c2.reserve(idx.size());
    for(int i: idx){ b2.push_back(all_boxes[i]); s2.push_back(all_scores[i]); c2.push_back(all_classes[i]); }
    all_boxes.swap(b2); all_scores.swap(s2); all_classes.swap(c2);
  }

  // ---- 4) NMS：class-agnostic 或 class-aware :contentReference[oaicite:14]{index=14}
  std::vector<int> keep;
  if (person_only || !class_aware_nms){
    keep = nms_xyxy(all_boxes, all_scores, iou_thres);
  } else {
    // class-aware：每类分别做 NMS
    std::unordered_map<int, std::vector<int>> byc;
    for (int i=0;i<(int)all_classes.size();i++) byc[all_classes[i]].push_back(i);

    std::vector<int> keep_all;
    for (auto &kv : byc){
      std::vector<Box> b; b.reserve(kv.second.size());
      std::vector<float> s; s.reserve(kv.second.size());
      for(int idx : kv.second){ b.push_back(all_boxes[idx]); s.push_back(all_scores[idx]); }
      auto k2 = nms_xyxy(b, s, iou_thres);
      for(int j : k2) keep_all.push_back(kv.second[j]);
    }
    std::sort(keep_all.begin(), keep_all.end(), [&](int a,int b){ return all_scores[a]>all_scores[b]; });
    keep.swap(keep_all);
  }

  // ---- 5) 输出 numpy arrays ----
  const int M = (int)keep.size();
  auto out_boxes   = py::array_t<float>({M,4});
  auto out_classes = py::array_t<int>({M});
  auto out_scores  = py::array_t<float>({M});

  auto bbuf = out_boxes.mutable_unchecked<2>();
  auto cbuf = out_classes.mutable_unchecked<1>();
  auto sbuf = out_scores.mutable_unchecked<1>();

  for(int i=0;i<M;i++){
    int k = keep[i];
    bbuf(i,0)=all_boxes[k].x1;
    bbuf(i,1)=all_boxes[k].y1;
    bbuf(i,2)=all_boxes[k].x2;
    bbuf(i,3)=all_boxes[k].y2;
    cbuf(i)=all_classes[k];
    sbuf(i)=all_scores[k];
  }
  return py::make_tuple(out_boxes, out_classes, out_scores);
}

PYBIND11_MODULE(yolov8_postprocess_ext, m) {
  m.doc() = "YOLOv8 RKNN postprocess (DFL decode + NMS) in C++";
  m.def("yolov8_rknn_post_process", &yolov8_rknn_post_process_cpp,
        py::arg("outputs"),
        py::arg("input_w"),
        py::arg("input_h"),
        py::arg("num_classes") = 1,
        py::arg("conf_thres") = 0.25f,
        py::arg("iou_thres") = 0.45f,
        py::arg("topk") = 300,
        py::arg("person_only") = false,
        py::arg("person_id") = 0,
        py::arg("class_aware_nms") = false
  );
}
