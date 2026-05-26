import streamlit as st
import pandas as pd
import numpy as np
from tabs import tab1_summary, tab2_norm, tab3_capability, tab4_spc

st.set_page_config(layout="wide", page_title="스마트제조 공정품질 분석 시스템")

st.title("🏭 스마트제조 공정능력분석 & 통계적 공정관리(SPC) 플랫폼")
st.write("강의록 표준 설계 알고리즘 기반 데이터 통합 분석 및 실시간 리포팅 시스템")

# --- 1. 사이드바: 데이터 로드 및 공통 규격 설정 패널 ---
st.sidebar.header("📁 데이터 분석 조건 제어반")

@st.cache_data
def load_lecture_pvc_data():
    """강의록 08_공정능력분석 9~10페이지 PVC 점도 실습 데이터셋 원본 복구"""
    data = np.array([
        [3576.27, 3630.12, 3576.27, 3630.12, 3355.69, 3363.62],
        [3504.17, 3514.52, 3747.43, 3666.15, 3709.25, 3317.28],
        [3440.11, 3494.35, 3962.93, 3514.30, 3273.57, 3336.20],
        [3638.33, 3719.84, 3617.47, 3450.17, 3378.70, 3475.50],
        [3661.94, 3485.53, 3499.43, 3605.53, 3390.29, 3519.26]
    ])
    df_wide = pd.DataFrame(data, columns=['pl_1', 'pl_2', 'pl_3', 'pl_4', 'pl_5', 'pl_6'])
    return df_wide.melt(var_name='prod_line', value_name='viscocity')

uploaded_file = st.sidebar.file_uploader("CSV 혹은 엑셀 파일 업로드", type=["csv", "xlsx"])
df_raw = pd.read_csv(uploaded_file) if uploaded_file else load_lecture_pvc_data()

all_cols = df_raw.columns.tolist()

# 셀렉트박스 설명 한글화 보강
sg_col = st.sidebar.selectbox("부분군(Subgroup) 식별 구분 컬럼 선택 (예: prod_line)", all_cols, index=0)
val_col = st.sidebar.selectbox("계측 데이터 특성치(Value) 컬럼 선택 (예: viscosity)", all_cols, index=1 if len(all_cols) > 1 else 0)

# [단기변동 산출 수식 라디오 버튼 한글화 전환]
sigma_method = st.sidebar.radio(
    "단기 변동(σ_within) 산출 수식 조절", 
    ["합동 표준편차 방식 (Pooled Standard Deviation)", "부분군 범위 평균 방식 (R-bar 방식)"]
)

# 계측 컬럼 유효성 사전 검사 (잘못된 선택 시 에러 원천 차단 유도 장치)
if not pd.api.types.is_numeric_dtype(df_raw[val_col]):
    st.sidebar.error("❌ 알림: 특성치 컬럼에 문자가 선택되었습니다.")
    st.error(f"⚠️ **[분석 조건 설정 오류]** 현재 특성치 컬럼으로 숫자가 아닌 문자열 컬럼(`{val_col}`)이 선택되어 분석을 시작할 수 없습니다.")
    st.info("💡 **올바른 유도 가이드:** 왼쪽 제어반에서 **'계측 데이터 특성치 컬럼 선택'** 상자를 숫자가 들어있는 **`viscocity`**로 변경해 주십시오.")
    st.stop() # 하단 탭 연산을 구동하지 않고 멈춤

# 정상 선택 시 규격 상하한 계산 진행
default_mean = float(df_raw[val_col].mean())
default_target = 3500.0 if val_col == 'viscocity' else round(default_mean, 2)
default_tol = 500.0 if val_col == 'viscocity' else round(df_raw[val_col].std() * 3, 2)

st.sidebar.subheader("🎯 공정 제어 한계 규격(Specification)")
target_val = st.sidebar.number_input("고객 품질 목표값(Target)", value=default_target)
tolerance = st.sidebar.number_input("허용 공차 한계폭(Tolerance)", value=default_tol)

lsl = target_val - tolerance
usl = target_val + tolerance

st.sidebar.markdown(f"**확정 규격 상/하한값**")
st.sidebar.info(f"🔴 **USL (규격상한)**: {usl:.2f} \n\n🔵 **LSL (규격하한)**: {lsl:.2f}")

# --- 2. 인터랙티브 메인 탭 구조 라우팅 ---
t1, t2, t3, t4 = st.tabs([
    "🔍 1단계: 데이터 구조 요약", 
    "📊 2단계: 정규성 적합 검증", 
    "📈 3단계: 공정능력평가(Cp/Cpk)", 
    "📉 4단계: SPC 제어 관리도"
])

with t1: tab1_summary.render(df_raw, sg_col, val_col)
with t2: tab2_norm.render(df_raw, val_col)
with t3: tab3_capability.render(df_raw, sg_col, val_col, lsl, usl, sigma_method)
with t4: tab4_spc.render(df_raw, sg_col, val_col)
