# -*- coding: utf-8 -*-
from __future__ import annotations

import socket
import threading


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


class TcpServer:
    def __init__(self, router, roi_service):
        self.router = router
        self.roi_service = roi_service

    def serve_forever(
        self,
        bind: str,
        port: int,
        client_timeout: int,
        shm_path_for_log: str,
        model_path: str,
        restart_unit: str,
    ):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind, int(port)))
        srv.listen(5)
        srv.settimeout(1.0)

        print(f"[TCP] listen {bind}:{port} (shm={shm_path_for_log})", flush=True)
        print(f"[UPG] model_path={model_path} restart_unit={restart_unit}", flush=True)

        def client_worker(conn: socket.socket, addr):
            conn.settimeout(int(client_timeout))
            cid, online = self.roi_service.on_client_connected()
            self.roi_service.update_indicators()
            print(f"[TCP] client#{cid} {addr} connected (online={online})", flush=True)

            try:
                while True:
                    if self.roi_service.stopped:
                        break

                    cmd = recv_exact(conn, 3)
                    ok = self.router.dispatch(cmd, conn, self.roi_service)

                    # unknown cmd -> 正常断开（与你选择的“第一个方案”一致）
                    if not ok:
                        break

            except socket.timeout:
                pass
            except Exception as e:
                # 调试期如果你要更详细，直接加 traceback.print_exc()
                print(f"[TCP] client#{cid} {addr} closed: {e}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                online2 = self.roi_service.on_client_disconnected()
                self.roi_service.update_indicators()
                print(f"[TCP] client#{cid} {addr} disconnected (online={online2})", flush=True)

        while True:
            if self.roi_service.stopped:
                break
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=client_worker, args=(conn, addr), daemon=True).start()