from ndif_citations import events
def test_emit_noop_without_sink():
    events.clear_sink(); events.emit("done")  # must not raise
def test_emit_to_sink():
    got = []
    events.set_sink(got.append)
    events.emit("item_step", stage="process", title="X", step="summary")
    events.clear_sink()
    assert got[0].type == "item_step" and got[0].stage == "process"
    assert got[0].data["title"] == "X" and got[0].data["step"] == "summary"
def test_sink_is_thread_local():
    import threading
    main_got, thread_got = [], []
    events.set_sink(main_got.append)
    def worker():
        events.emit("log", msg="bg")           # no sink in this thread
        events.set_sink(thread_got.append); events.emit("log", msg="bg2"); events.clear_sink()
    t = threading.Thread(target=worker); t.start(); t.join()
    events.emit("log", msg="main"); events.clear_sink()
    assert [e.data["msg"] for e in main_got] == ["main"]
    assert [e.data["msg"] for e in thread_got] == ["bg2"]
