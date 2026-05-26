import streamlit as st
import pandas as pd
import numpy as np
from tabs import tab1_summary, tab2_norm, tab3_capability, tab4_spc, tab5_attribute

st.set_page_config(layout="wide", page_title="스마트품질 공정분석 종합 대시보드")

st.title("🏭 스마트제조 공정능력분석 & 통계적 공정관리(SPC) 플랫폼")
st.write("강의록 표준 알고리즘 기반 입체적 품질 가시화 및 실시간 데이터 리포팅 시스템")

# --- 1. 사이드바: 글로벌 데이터 업로드 및 제어반 ---
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
sg_col = st.sidebar.selectbox("부분군(Subgroup) 식별 구분 컬럼 선택 (예: prod_line)", all_cols, index=0)
val_col = st.sidebar.selectbox("계측 데이터 특성치(Value) 컬럼 선택 (예: viscocity)", all_cols, index=1 if len(all_cols) > 1 else 0)

# [한글 방어 유도 장치] 특성치에 글자 들어가면 즉시 안전 차단
if not pd.api.types.is_numeric_dtype(df_raw[val_col]):
    st.sidebar.error("❌ 오류: 특성치에 문자가 지정됨")
    st.error(f"⚠️ **[분석 조건 설정 오류]** 현재 특성치 컬럼으로 숫자가 아닌 문자열 컬럼(`{val_col}`)이 선택되어 분석을 진행할 수 없습니다.")
    st.info("💡 **올바른 가이드:** 왼쪽 제어반에서 **'계측 데이터 특성치 컬럼 선택'** 상자를 숫자가 들어있는 **`viscocity`**로 변경해 주십시오.")
    st.stop()

# 산출 수식 및 규격 설정 (사이드바 컴포넌트 유지)
st.session_state['sigma_method'] = st.sidebar.radio(
    "단기 변동(σ_within) 산출 수식 선택 (강의록 6p)", 
    ["합동 표준편차 방식 (Pooled Standard Deviation)", "부분군 범위 평균 방식 (R-bar 방식)"]
)

default_mean = float(df_raw[val_col].mean())
default_target = 3500.0 if val_col == 'viscocity' else round(default_mean, 2)
default_tol = 500.0 if val_col == 'viscocity' else round(df_raw[val_col].std() * 3, 2)

st.sidebar.subheader("🎯 공정 제어 한계 규격(Specification)")
target_val = st.sidebar.number_input("고객 품질 목표값(Target)", value=default_target)
tolerance = st.sidebar.number_input("허용 공차 한계폭(Tolerance)", value=default_tol)

st.session_state['lsl'] = target_val - tolerance
st.session_state['usl'] = target_val + tolerance

st.sidebar.markdown(f"**확정 규격 상/하한값**")
st.sidebar.info(f"🔴 **USL (규격상한)**: {st.session_state['usl']:.2f} \n\n🔵 **LSL (규격하한)**: {st.session_state['lsl']:.2f}")

# --- 2. 5대 시각화 대시보드 탭 배치 (인자를 무조건 3개로 통일하여 TypeError 영구 소멸) ---
t1, t2, t3, t4, t5 = st.tabs([
    "🔍 1단 대시보드: 데이터 요약", 
    "📊 2단 대시보드: 정규성 분포 비교", 
    "📈 3단 대시보드: 라인별 공정능력", 
    "📉 4단 대시보드: 계량형 SPC 관리도",
    "🏷️ 5단 대시보드: 계수형 불량 관리도"
])

with t1: tab1_summary.render(df_raw, sg_col, val_col)
with t2: tab2_norm.render(df_raw, sg_col, val_col)
with t3: tab3_capability.render(df_raw, sg_col, val_col)
with t4: tab4_spc.render(df_raw, sg_col, val_col)
with t5: tab5_attribute.render(df_raw, sg_col, val_col)
