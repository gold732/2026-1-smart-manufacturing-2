import streamlit as st
import pandas as pd
import numpy as np
# 분리된 개별 탭 모듈 호출
from tabs import tab1_summary, tab2_norm, tab3_capability, tab4_spc

st.set_page_config(layout="wide", page_title="스마트제조 공정품질 분석 시스템")

st.title("🏭 스마트제조 공정능력분석 & 통계적 공정관리(SPC) 플랫폼")
st.write("강의록 표준 설계 알고리즘 기반 데이터 통합 분석 및 실시간 리포팅 시스템")

# --- 1. 사이드바: 글로벌 데이터 로드 및 공통 규격 설정 패널 ---
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
sg_col = st.sidebar.selectbox("부분군(Subgroup) 식별 구분 컬럼 선택", all_cols, index=0)
val_col = st.sidebar.selectbox("계측 데이터 특성치(Value) 컬럼 선택", all_cols, index=1 if len(all_cols) > 1 else 0)

# 가이드 목표/규격 경계치 입력 인터페이스
default_mean = float(df_raw[val_col].mean())
default_target = 3500.0 if val_col == 'viscocity' else round(default_mean, 2)
default_tol = 500.0 if val_col == 'viscocity' else round(df_raw[val_col].std() * 3, 2)

st.sidebar.subheader("🎯 공정 제어 한계 규격(Specification)")
target_val = st.sidebar.number_input("고객 품질 목표값(Target)", value=default_target)
tolerance = st.sidebar.number_input("허용 공차 한계폭(Tolerance)", value=default_tol)

lsl = target_val - tolerance
usl = target_val + tolerance

st.sidebar.markdown(f"**확정 규격 상/하한값**")
st.sidebar.info(f"🔴 **USL**: {usl:.2f} \n\n🔵 **LSL**: {lsl:.2f}")

# --- 2. 인터랙티브 메인 탭 구조 라우팅 ---
t1, t2, t3, t4 = st.tabs([
    "🔍 1단계: 데이터 구조 요약", 
    "📊 2단계: 정규성 적합 검증", 
    "📈 3단계: 공정능력평가(Cp/Cpk)", 
    "📉 4단계: SPC 제어 관리도"
])

# 각 탭에 독립된 전담 모듈 함수를 바인딩하여 렌더링
with t1:
    tab1_summary.render(df_raw, sg_col, val_col)

with t2:
    tab2_norm.render(df_raw, val_col)

with t3:
    tab3_capability.render(df_raw, sg_col, val_col, lsl, usl)

with t4:
    tab4_spc.render(df_raw, sg_col, val_col)
