"""ROI 红框标定:摄像头实时画面上拖拽画框,保存到 roi_regions.json。
合成数据时 marker 中心限制在这些框内(框外是固定干扰区)。

操作:
  鼠标左键拖拽 = 画一个红框
  u = 撤销最后一个框
  c = 清空所有框
  s = 保存并退出
  q/ESC = 不保存退出
"""
import json
from pathlib import Path
import cv2, numpy as np

SETTINGS = Path("roi_regions.json")


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("无法打开摄像头"); return 1

    boxes = []                      # [x,y,w,h]
    if SETTINGS.exists():
        boxes = json.loads(SETTINGS.read_text(encoding="utf-8")).get("boxes", [])
        print(f"已加载 {len(boxes)} 个现有框")
    drag = {"on": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag.update(on=True, x0=x, y0=y, x1=x, y1=y)
        elif ev == cv2.EVENT_MOUSEMOVE and drag["on"]:
            drag.update(x1=x, y1=y)
        elif ev == cv2.EVENT_LBUTTONUP and drag["on"]:
            drag["on"] = False
            x0, y0, x1, y1 = drag["x0"], drag["y0"], x, y
            bx, by = min(x0, x1), min(y0, y1)
            bw, bh = abs(x1 - x0), abs(y1 - y0)
            if bw > 10 and bh > 10:
                boxes.append([bx, by, bw, bh])

    cv2.namedWindow("ROI 标定")
    cv2.setMouseCallback("ROI 标定", on_mouse)
    print("拖拽画框 | u撤销 | c清空 | s保存退出 | q放弃退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, str(i), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if drag["on"]:
            cv2.rectangle(frame, (drag["x0"], drag["y0"]), (drag["x1"], drag["y1"]), (0, 200, 255), 1)
        cv2.putText(frame, f"boxes={len(boxes)} | drag=draw u=undo c=clear s=save q=quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("ROI 标定", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('s'),):
            SETTINGS.write_text(json.dumps({"frame": [1280, 720], "boxes": boxes},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"已保存 {len(boxes)} 个框 -> {SETTINGS.resolve()}")
            break
        elif k in (ord('q'), 27):
            print("放弃,未保存"); break
        elif k == ord('u') and boxes:
            boxes.pop()
        elif k == ord('c'):
            boxes.clear()
    cap.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
