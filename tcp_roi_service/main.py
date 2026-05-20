# -*- coding: utf-8 -*-
from __future__ import annotations

from config.load import load_config

from adapters.gpio import IndicatorController
from adapters.shm_reader import ShmReader
from adapters.conf_shm import ConfShmWriter

from app.roi_service import RoiService
from app.command_router import CommandRouter
from app.tcp_server import TcpServer

from app.handlers import set_roi, set_poly, set_mask, set_thr, ask, upg


def main():
    cfg = load_config()

    # adapters
    gpio = IndicatorController(beep_on_err=True)
    gpio.init()

    shm = ShmReader(cfg.shm_path)
    confw = ConfShmWriter(cfg.conf_shm)

    # service（参数对齐你的 RoiAlarmServer 构造）
    svc = RoiService(
        shm_reader=shm,
        conf_writer=confw,
        indicators=gpio,
        model_path=cfg.model_path,
        restart_unit_name=cfg.restart_unit,
        default_sensitivity=cfg.default_sensitivity,
        poll_ms=cfg.poll_ms,
        max_alarm=cfg.max_alarm,
        msk_eps=cfg.msk_eps,
        print_shm=cfg.print_shm,
        print_roi=cfg.print_roi,
        print_thr=cfg.print_thr,
        print_msk=cfg.print_msk,
    )

    # router
    router = CommandRouter()
    router.register(b"SET", set_roi.handle)
    router.register(b"POL", set_poly.handle)
    router.register(b"MSK", set_mask.handle)
    router.register(b"THR", set_thr.handle)
    router.register(b"ASK", ask.handle)
    router.register(b"UPG", upg.handle)

    # tcp
    server = TcpServer(router=router, roi_service=svc)
    server.serve_forever(
        bind=cfg.bind,
        port=cfg.port,
        client_timeout=cfg.client_timeout,
        shm_path_for_log=cfg.shm_path,
        model_path=cfg.model_path,
        restart_unit=cfg.restart_unit,
    )


if __name__ == "__main__":
    main()