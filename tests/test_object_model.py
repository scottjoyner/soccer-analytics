from soccer_edge.object_model import object_rows_from_result


class FakeBoxes:
    xyxy = [[1, 2, 3, 4]]
    conf = [0.9]
    cls = [0]


class FakeResult:
    names = {0: "player"}
    boxes = FakeBoxes()


def test_object_rows_from_result() -> None:
    rows = object_rows_from_result(FakeResult())
    assert rows[0]["class_name"] == "player"
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["x1"] == 1.0
