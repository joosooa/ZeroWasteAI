import os
import json
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# 환경변수 로드
load_dotenv()

app = FastAPI()

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

# =====================================================================
# 1. API 키 및 클라이언트 설정
# =====================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
CUSTOM_API_URL = os.getenv("CUSTOM_API_URL")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"

# Supabase 연동 방어 로직
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        supabase = None
        print("[WARN] Supabase 환경변수가 설정되지 않았습니다.")
except Exception as e:
    supabase = None
    print(f"[ERROR] Supabase 클라이언트 생성 실패: {e}")

# CJOneFlow LLM API 연동 방어 로직
if LLM_BASE_URL:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    print(f"[INFO] LLM 클라이언트 연결 시도: {LLM_BASE_URL}")
else:
    print("[ERROR] LLM_BASE_URL이 설정되지 않았습니다.")

# =====================================================================
# 2. Pydantic 데이터 모델 정의
# =====================================================================
class LoginRequest(BaseModel):
    store_id: str
    password: str

class TriggerRequest(BaseModel):
    store_id: str

class CouponRequest(BaseModel):
    store_id: str
    product_id: str


# =====================================================================
# 3. 데이터 조회 및 정제 헬퍼 함수
# =====================================================================
def get_inventory_from_db(store_id: str):
    """현재고, 당일생산량, 마스터정보를 조인하여 반환합니다."""
    if not supabase:
        return []
        
    try:
        # store_id를 DB 타입인 int(bigint)로 변환
        db_store_id = int(store_id)
        
        # 1. 현재 재고 조회
        inv_res = supabase.table("inventory").select("product_id, current_stock").eq("store_id", db_store_id).execute()
        if not inv_res.data:
            return []

        # 2. 당일 생산량 조회
        prod_res = supabase.table("daily_production").select("product_id, quantity").eq("store_id", db_store_id).execute()
        prod_dict = {str(item["product_id"]).strip().lower(): item["quantity"] for item in prod_res.data}

        # 3. 마스터 정보 조회
        master_res = supabase.table("products_master_with_keywords").select("product_id, name").execute()
        
        name_dict = {}
        if master_res.data:
            for item in master_res.data:
                p_id = str(item["product_id"]).strip().lower()
                name_dict[p_id] = item["name"]

        # 4. 데이터 병합
        combined_inventory = []
        for inv in inv_res.data:
            raw_id = str(inv["product_id"]).strip()
            lookup_id = raw_id.lower()
            name = name_dict.get(lookup_id, f"미등록 상품({raw_id[:6]})")

            combined_inventory.append({
                "product_id": raw_id,
                "name": name,
                "daily_production": str(prod_dict.get(lookup_id, 0)), 
                "stock": inv["current_stock"] 
            })
            
        return combined_inventory
        
    except ValueError:
        print(f"[ERROR] 점포 번호({store_id})는 숫자여야 합니다.")
        return []
    except Exception as e:
        print(f"[ERROR] 데이터 병합 중 서버 오류 발생: {e}")
        return []

# 판매량 예측을 위해 매장 위치 기반 실시간 날씨 데이터를 가져오는 함수 => custom API의 Input 중 하나
def get_real_weather(store_id: str):
    """매장 위치를 기반으로 실시간 날씨를 가져옵니다."""
    lat = 37.5665
    lon = 126.9780 
    
    if supabase:
        try:
            # store_id를 int로 변환
            res = supabase.table("stores").select("lat, lng").eq("id", int(store_id)).execute()
            if res.data:
                lat = float(res.data[0]["lat"])
                lon = float(res.data[0]["lng"])
        except Exception as e:
            print(f"[WARN] 매장 좌표 조회 실패: {e}")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, 
        "longitude": lon, 
        "current": "temperature_2m,relative_humidity_2m,precipitation,snowfall,shortwave_radiation", 
        "timezone": "Asia/Seoul"
    }
    
    try:
        w_res = requests.get(url, params=params, timeout=100)
        w_res.raise_for_status()
        d = w_res.json().get("current", {})
        return {
            "temp": d.get("temperature_2m", 0.0), 
            "precip": d.get("precipitation", 0.0), 
            "hum": d.get("relative_humidity_2m", 0.0), 
            "solar": d.get("shortwave_radiation", 0.0), 
            "snow": d.get("snowfall", 0.0)
        }
    except Exception:
        return {"temp": 15.5, "precip": 0.0, "hum": 45.0, "solar": 12.0, "snow": 0.0}

# =====================================================================
# 4. CJOneFlowLLM CUstom API 예측 파이프라인
# 실제 환경에서는 특정 시간(오후 5시)에 자동으로 실행되지만, 현재는 데모 버전으로 수동 버튼 트리거로 구현되어 있습니다.
# =====================================================================

def run_prediction_pipeline(store_id: str):
    """현재 매장 재고와 날씨 데이터를 종합하여 실제 CJOneFlow Custom API를 통해 당일 폐기율을 예측합니다."""
    print(f"\n[START] {store_id} 점포 폐기율 예측 파이프라인 가동 (CJOneFlow Custom API 연동)")
    try:
        current_inventory = get_inventory_from_db(store_id)
        current_weather = get_real_weather(store_id)

        api_inventory = []
        for item in current_inventory:
            if "미등록" in item["name"]: 
                continue
            api_inventory.append({
                "product_id": item["product_id"], 
                "daily_production": int(item["daily_production"]), 
                "stock": int(item["stock"])
            })

        if not api_inventory: 
            return {"status": "error", "message": "분석할 데이터가 없습니다."}

        # Custom API 호출 로직
        core_data = {
            "store_id": str(store_id), 
            "current_inventory": api_inventory, 
            "closing_time": 24, 
            "weather": current_weather
        }
        
        payload = {"paths": [""], "config": core_data}
        headers = {"Content-Type": "application/json", "x-api-key": CUSTOM_API_KEY}
        
        response = requests.post(CUSTOM_API_URL, json=payload, headers=headers, timeout=10000)
        response.raise_for_status()
        
        data_layer = response.json().get('data', {})
        actual_res = data_layer.get('data', data_layer) 
        
        waste_rate = actual_res.get('overall_waste_rate', 0.0)
        is_trigger = actual_res.get('trigger_coupon', False)
        
        target_product_id = ""
        details = actual_res.get("details", [])
        if details:
            worst_item = max(details, key=lambda x: x.get("waste_qty", 0))
            target_product_id = worst_item.get("product_id", "")
        
        return {
            "status": "success", 
            "waste_rate": round(float(waste_rate), 1), 
            "trigger_coupon": bool(is_trigger), 
            "target_product_id": target_product_id
        }
    except Exception as e:
        print(f"[ERROR] 파이프라인 내부 에러 발생: {e}")
        return {"status": "error", "message": "예측 파이프라인 처리 중 오류가 발생했습니다."}

# =====================================================================
# 4. CJOneFlow LLM API를 활용한 마케팅 문구 생성 로직
# =====================================================================
def generate_llm_push(product_name: str, keywords: str, discount: int, store_name: str):
    """
    [프롬프트 핵심 전략] 
    전체 품목 할인 쿠폰임을 강조하되, 폐기 위험이 가장 높은 상품을 '미끼(Hook)'로 사용하여 식욕을 자극합니다.
    """
    system_prompt = f"""
    너는 뚜레쥬르 점주를 대신해 고객의 매장 방문을 유도하는 전문 마케터야.
    반드시 다음 JSON 형식으로만 답변해: {{"headline": "...", "action_line": "...", "push_text": "..."}}

    [문구 작성 핵심 규칙]
    1. headline (헤드라인)
       - "마감 할인 쿠폰" 또는 "당일 한정 할인 쿠폰"이라는 문구를 메인으로 써.
       - 오늘까지만 쓸 수 있다는 혜택의 한정성을 강조해.
       - [중요 금지사항]: 헤드라인에 특정 빵 이름과 할인을 직접 연결하지 마. (예: "소보로빵 20% 할인" -> 절대 금지. 이 쿠폰은 매장 전체 빵에 적용됨)

    2. push_text (본문 텍스트)
       - 타겟 상품([추천 빵])을 미끼(Hook)로 던져서 고객의 식욕을 자극해.
       - [키워드]를 활용해 자연스럽고 맛깔나게 묘사해줘. (예시: "달달하고 부드러운 [추천 빵]이 땡기는 저녁, ~")
       - 내가 제시해준 [키워드]와 [추천 빵] 외에 다른 새로운 상품을 언급하지 마.
       - "폐기", "재고 처리" 등 매장의 내부 사정은 절대 언급하지 마.
       - "오늘 [매장명]에서 만나요!"라는 뉘앙스로 유도해.
       - 절대 뚜레쥬르에서 사용할 수 있는 쿠폰이 아니라, 특정 매장에만 적용된다는 점을 강조해줘. (예: "뚜레쥬르 [매장명]점에서만 쓸 수 있는 깜짝 쿠폰으로 ~")
       - 문장은 깔끔하고 간결하게 작성해줘. 최대 2문장 이상 넘어가지 않도록 해.

    3. action_line (버튼 문구)
       - 고객이 당장 누르고 싶어지는 짧은 행동 유도 문구 (예: "쿠폰 받고 빵 고르러 가기", "오늘의 혜택 확인하기")
    """
    
    user_content = f"추천 빵: {product_name} / 키워드: {keywords} / 할인율: {discount}% / 매장명: {store_name}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_content}
            ],
            timeout=100
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"): 
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        # LLM 에러 시 기본으로 나갈 백업 텍스트
        return {
            "headline": f"⏰ 당일 한정 마감 세일! {discount}% 할인 쿠폰 도착", 
            "action_line": "쿠폰 받고 빵 고르러 가기", 
            "push_text": f"{keywords} {product_name} 어떠세요? 오늘 저녁에만 쓸 수 있는 깜짝 쿠폰으로 뚜레쥬르에서 맛있는 빵을 알뜰하게 즐겨보세요!"
        }

# =====================================================================
# 5. API 라우터
# =====================================================================
@app.post("/api/login")
def login(req: LoginRequest):
    if req.password == "0000":
        store_name_clean = "UNKNOWN"
        if supabase:
            try:
                res = supabase.table("stores").select("store_name_clean").eq("id", int(req.store_id)).execute()
                if res.data:
                    store_name_clean = res.data[0].get("store_name_clean", "UNKNOWN")
            except ValueError:
                raise HTTPException(status_code=400, detail="점포 번호는 숫자여야 합니다.")
            except Exception as e:
                print(f"[WARN] store_name_clean 조회 실패: {e}")
                
        return {"status": "success", "store_id": req.store_id, "store_name_clean": store_name_clean}
    raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

@app.get("/api/inventory/{store_id}")
def get_inventory(store_id: str):
    return {"inventory": get_inventory_from_db(store_id)}

@app.post("/api/trigger-test")
def manual_trigger(req: TriggerRequest):
    return run_prediction_pipeline(req.store_id)

@app.post("/api/issue-coupon")
def issue_coupon(req: CouponRequest):
    print(f"\n[START] 쿠폰 발행 및 고객 필터링 프로세스 시작 (Store: {req.store_id})")
    
    target_item = "베스트 상품"
    taste_raw = ""
    store_name = f"뚜레쥬르 {req.store_id}점"
    city = "서울"
    dong = "을지로동"
    
    if supabase:
        try:
            prod_res = supabase.table("products_master_with_keywords").select("name, taste_keywords").eq("product_id", req.product_id).execute()
            if prod_res.data:
                target_item = prod_res.data[0].get("name", "베스트 상품")
                taste_raw = prod_res.data[0].get("taste_keywords", "")
        except Exception as e:
            print(f"[WARN] 마스터 정보 조회 에러: {e}")

        try:
            store_res = supabase.table("stores").select("region_l1, region_l3, store_name_clean").eq("id", int(req.store_id)).execute()
            if store_res.data:
                city = store_res.data[0].get("region_l1", "서울")
                dong = store_res.data[0].get("region_l3", "을지로동")
                store_name = store_res.data[0].get("store_name_clean", f"뚜레쥬르 {dong}점")
        except Exception as e:
            print(f"[WARN] 매장 DB 조회 에러: {e}")

    keywords_list = [k.strip() for k in taste_raw.split(',')] if taste_raw else ["맛있는", "신선한"]
    selected_keywords = ", ".join(keywords_list[:2])

    customers = []
    if supabase:
        try:
            cust_res = supabase.table("sales_history").select("customer_id, sales_time").eq("region_l1", city).eq("region_l3", dong).execute()
            if cust_res.data:
                for row in cust_res.data:
                    customers.append({
                        "id": row.get("customer_id", "UNKNOWN"),
                        "visit_hour": int(row.get("sales_time", 12)) 
                    })
        except Exception as e:
            print(f"[WARN] 고객 DB 조회 에러: {e}")

    # 시간대별 고객 그룹화
    group_20, group_15, group_10 = [], [], []
    for c in customers:
        hour = c.get("visit_hour", 12)
        if hour >= 17: 
            group_20.append(c["id"])
        elif 14 <= hour < 17: 
            group_15.append(c["id"])
        else: 
            group_10.append(c["id"])

    groups_to_process = [
        {"discount": 20, "label": "17시 이후 방문 고객", "users": group_20},
        {"discount": 15, "label": "14~17시 방문 고객", "users": group_15},
        {"discount": 10, "label": "14시 이전 방문 고객", "users": group_10},
    ]

    final_results = []
    db_insert_payload = []

    print(f"[INFO] 타겟 그룹별 LLM 맞춤 마케팅 문구 생성을 시작합니다.")
    
    for g in groups_to_process:
        if not g["users"]: 
            continue
        
        llm_reply = generate_llm_push(target_item, selected_keywords, g["discount"], store_name)
        
        for uid in g["users"]:
            db_insert_payload.append({
                "store_id": req.store_id,
                "customer_id": uid,
                "product_id": req.product_id,
                "discount_rate": g["discount"],
                "headline": llm_reply.get("headline", ""),
                "push_text": llm_reply.get("push_text", ""),
                "action_line": llm_reply.get("action_line", ""),
                "status": "pending",  
                "created_at": datetime.now().isoformat()
            })
            
        final_results.append({
            "discount": g["discount"],
            "label": g["label"],
            "count": len(g["users"]),
            "copy": llm_reply
        })

    # Supabase push_queue 테이블 적재
    if supabase and db_insert_payload:
        try:
            supabase.table("push_queue").insert(db_insert_payload).execute()
            print(f"[SUCCESS] 총 {len(db_insert_payload)}건의 푸시 예약 데이터 저장 완료")
        except Exception as e:
            print(f"[ERROR] push_queue 저장 실패: {e}")

    return {"status": "success", "results": final_results}


# =====================================================================
# 6. 프론트엔드 서빙
# =====================================================================
@app.get("/", response_class=FileResponse)
def serve_dashboard():
    """같은 폴더에 있는 index.html 파일을 찾아서 웹 브라우저에 보여줍니다."""
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)