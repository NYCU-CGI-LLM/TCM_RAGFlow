#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://127.0.0.1:9380/api/v1"
API_KEY="ragflow-c1NzBmZjdhYTM1MDExZjBhMjlhMzdkMj"
DATASET_NAME=${1:-test_4_v2}

echo "Resolving dataset name '${DATASET_NAME}' to ID..."
DATASET_JSON="$(
  curl -sS -G \
    -H "Authorization: Bearer ${API_KEY}" \
    "${API_BASE}/datasets" \
    --data-urlencode "page=1" \
    --data-urlencode "page_size=1" \
    --data-urlencode "name=${DATASET_NAME}" \
    --data-urlencode "orderby=create_time" \
    --data-urlencode "desc=true"
)"

if [[ -z "${DATASET_JSON}" ]]; then
  echo "Received empty response while looking up dataset '${DATASET_NAME}'." >&2
  exit 1
fi

if ! DATASET_ID=$(
  DATASET_JSON="${DATASET_JSON}" python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("DATASET_JSON", "")
try:
    payload = json.loads(raw)
except json.JSONDecodeError as exc:
    sys.stderr.write(f"Failed to parse dataset lookup response: {exc}\n")
    sys.exit(1)

if payload.get("code") != 0:
    sys.stderr.write(f"Dataset lookup failed with code={payload.get('code')}: {payload.get('message')}\n")
    sys.exit(1)

data = payload.get("data") or []
if not data:
    sys.stderr.write("Dataset lookup returned no results.\n")
    sys.exit(1)

dataset_id = data[0].get("id")
print(dataset_id)
PY
); then
  exit 1
fi

if [[ -z "${DATASET_ID:-}" ]]; then
  echo "Failed to resolve dataset name '${DATASET_NAME}'."
  exit 1
fi

echo "Using dataset ID: ${DATASET_ID}"

curl -X POST "${API_BASE}/retrieval_simple_rag/${DATASET_ID}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "患者临床信息：\n主诉：反复胸闷憋喘2年余，加重1天\n现病史：患者2年前开始反复出现胸闷憋喘，活动后多发，休息后可缓解，未予重视，2月前患者胸闷憋喘较前加重，遂于2020年06月至徐州铜山区利国医院就诊，查生化示：ALT 362U/L,AST 566U/L,CREA 120mmol/L,UA 545umol/L,CK 404U/L,CKMB 29U/L,TNI 0.336ng/mL,BNP＞5000pg/mL。心脏超声：1.左房、左室增大，伴二尖瓣中量返流；2.主动脉瓣增厚、局部钙化伴少量返流；3.肺动脉瓣少量返流；4.三尖瓣少量返流；5.左心功能差 EF12%；6.心包少量积液声像。双下肢超声：双下肢动脉硬化伴斑块形成，双下肢深静脉瓣功能不全；双下肢肌肉间静脉增宽，内血流瘀滞可能性大，左侧足背动脉血流速度缓慢，双小腿皮下软组织水肿声像。诊断考虑肺部感染、急性冠脉综合征，心力衰竭，高血压病，肝损伤，肾功能异常，双下肢浅静脉血栓等，予抗血小板聚集、降脂、稳定斑块、利尿等治疗后，症状未见明显好转，患者及家属要求出院，出院后患者胸闷憋喘较前加重，端坐呼吸，咳嗽咳痰，伴双下肢水肿，2020年07月23遂至徐州医科大学附属医院急诊就诊，查hsTnt 169.8ng/L，查心电图提示室内传导阻滞，室性早搏，心脏超声：左室室壁运动幅度及增厚率明显减低；左心增大、二尖瓣少中量返流；三尖瓣少量返流，肺动脉高压（60mmHg）；主动脉瓣少量返流，主动脉瓣半增厚，局部钙化。房间隔小缺损或卵圆孔未闭；左心功能明显减低，微量心包积液，EF 27%。予左西孟旦、托拉塞米利尿、环磷腺苷葡胺静滴营养心肌、盐酸罗沙替丁护胃等治疗，建议转至ICU，择期行冠脉造影检查，患者及家属拒绝，长期予口服呋塞米、螺内酯、阿司匹林、沙库巴曲缬沙坦钠片治疗，1天前患者胸闷憋喘进一步加重，夜间憋醒，端坐位缓解，遂至我院就诊，为求进一步系统治疗，收住入院，入院时：患者胸闷憋喘，端坐呼吸，夜间不能平卧，无胸痛、放射痛，无心慌乏力，头晕偶作，无黑矇晕厥，无恶寒发热，无腹胀腹满，纳差，二便尚调，夜寐差。减低，微量心包积液，EF 27%。予左西孟旦、托拉塞米利尿、环磷腺苷葡胺静滴营养心肌、盐酸罗沙替丁护胃等治疗，建议转至ICU，择期行冠脉造影检查，患者及家属拒绝，长期予口服呋塞米、螺内酯、阿司匹林、沙库巴曲缬沙坦钠片治疗，1天前患者胸闷憋喘进一步加重，夜间憋醒，端坐位缓解，遂至我院就诊，为求进一步系统治疗，收住入院，入院时：患者胸闷憋喘，端坐呼吸，夜间不能平卧，无胸痛、放射痛，无心慌乏力，头晕偶作，无黑矇晕厥，无恶寒发热，无腹胀腹满，纳差，二便尚调，夜寐差。\n体格检查：神志清晰，精神尚可，形体适中，语言清晰，口唇发绀；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽黑紫，双下肢凹陷性水肿，舌淡紫，苔薄白，脉涩。\n\n请从以下选项中选择：\n气虚血瘀证、风热伤络证、心阳不振证、风寒痹阻证、肝肾亏虚证等"
      }
    ]
  }'
