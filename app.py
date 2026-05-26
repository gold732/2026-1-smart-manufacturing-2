import streamlit as st
import pandas as pd
import numpy as np
import stats_engine as engine
import visualizer as vis

st.set_page_config(layout="wide", page_title="스마트제조 공정품질 분석 시스템")

st.title("🏭 스마트제조 공정능력분석 & 통계적 공정관리(SPC) 웹 대시보드")
st.write("강의록 표준 설계 알고리즘 기반 데이터 통합 분석 및 실시간 리포팅 시스템")

# --- 1. 사이드바: 데이터 제어 및 조절 파라미터 설정 패널 ---
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

# [확장된 파라미터]: 단기 분산 계산 메커니즘을 유저가 직접 마우스 클릭으로 조절 선택
sigma_method = st.sidebar.radio(
    "단기 변동(σ_within) 산출 수식 조절", 
    ["Pooled Standard Deviation (합동표준편차)", "Subgroup Range Average (R-bar 방식)"]
)

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

# --- 2. 메인 데이터 레이아웃 화면 배정 ---
tab1, tab2 = st.tabs(["📊 1~3단계: 공정능력분석 보고서", "📉 4단계: 통계적 공정관리(SPC) 관리도"])

with tab1:
    st.header("🔍 1단계: 실시간 수집 공정 데이터 구조 분석")
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("데이터 데이터프레임 구조 요약")
        st.dataframe(df_raw.head(12), use_container_width=True)
        st.caption(f"전체 관측치 개수: {len(df_raw)}행 | 검측 분석 대상 변수명: {val_col}")
    with col2:
        fig_box = vis.plot_boxplot_dashboard(df_raw, sg_col, val_col)
        st.plotly_chart(fig_box, use_container_width=True)

    st.header("📊 2단계: 정규성(Normality Test) 적합 검증")
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    c_test1, c_test2 = st.columns([4, 6])
    with c_test1:
        st.metric(label="Shapiro-Wilk Test p-value", value=f"{p_value:.6f}")
        if is_normal:
            st.success("✅ 정규성 조건 만족: 원본 데이터 기반으로 즉시 가용 공정능력지수를 연산합니다.")
            final_df = df_raw.copy()
            final_lsl, final_usl = lsl, usl
        else:
            st.warning("⚠️ 정규성 조건 불만족 (p < 0.05): 강의록 지침에 의거하여 Box-Cox 거듭제곱 변환 알고리즘을 가동합니다.")
            if (df_raw[val_col] <= 0).any():
                st.error("❌ 에러: Box-Cox 변환은 양수 데이터 영역에만 처리가 지원됩니다. 데이터 적합성을 확인하십시오.")
                final_df = df_raw.copy()
                final_lsl, final_usl = lsl, usl
            else:
                transformed_vals, lsl_t, usl_t, lambda_val = engine.apply_box_cox(df_raw[val_col].values, lsl, usl)
                final_df = df_raw.copy()
                final_df[val_col] = transformed_vals
                final_lsl, final_usl = lsl_t, usl_t
                st.info(f"💡 최적 탐색 변환 Parameter Lambda (λ): {lambda_val:.4f}")
    with c_test2:
        fig_qq = vis.plot_qq_dashboard(final_df[val_col].values)
        st.plotly_chart(fig_qq, use_container_width=True)

    st.header("📈 3단계: 단기/장기 잠재 공정능력(Process Capability Analysis) 평가")
    metrics = engine.analyze_process_capability(final_df, sg_col, val_col, final_lsl, final_usl, method=sigma_method)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("단기 잠재력 (산포 중심) Cp", f"{metrics['Cp']:.4f}")
    m_col2.metric("단기 실효 능력 (치우침 고려) Cpk", f"{metrics['Cpk']:.4f}")
    m_col3.metric("장기 잠재력 (종합 변동) Pp", f"{metrics['Pp']:.4f}")
