// 解析与配对的纯逻辑。无 DOM 依赖，可 node 测试。

// 凑近对比的像素阈值（spec 5.3）。NEAR<FAR 形成滞回死区。
export const NEAR = 140;
export const FAR = 180;

// 用给定 catalog + mapping 造一个 resolver：symbolId -> 香水（或未映射降级对象）。
export function makeResolver(catalog, mapping) {
  return function resolve(symbolId) {
    const key = mapping.map[symbolId];
    if (!key || !catalog[key]) return { unregistered: true, symbolId };
    return { ...catalog[key], symbolId, key };
  };
}

// 滞回状态机：根据上一帧是否展开 + 当前距离，决定这一帧是否展开对比卡。
// d<NEAR 开；d>FAR 关；之间维持原状。
export function proximityState(wasOpen, distance) {
  if (distance < NEAR) return true;
  if (distance > FAR) return false;
  return wasOpen;
}
