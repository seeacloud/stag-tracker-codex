import sys


def test_asset_dir_points_to_table_app():
    from vision_fusion import tuio_table_app

    assert tuio_table_app.ASSET_DIR.name == "table_app"
    assert tuio_table_app.ASSET_DIR.is_dir()


def test_default_http_port_is_8778():
    from vision_fusion import tuio_table_app

    argv = sys.argv
    sys.argv = ["tuio_table_app"]
    try:
        args = tuio_table_app.parse_args()
    finally:
        sys.argv = argv
    assert args.http_port == 8778
    assert args.tuio_port == 3333


def test_reuses_watch_app_backend_symbols():
    from vision_fusion import tuio_table_app

    # 复用而非复制：这些名字应来自 tuio_watch_app
    assert hasattr(tuio_table_app, "SharedTuioState")
    assert hasattr(tuio_table_app, "make_handler")
    assert hasattr(tuio_table_app, "run_tuio_udp_listener")
