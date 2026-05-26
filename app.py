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
    m_col4.metric("장기 실효 능력 (종합 실효) Ppk", f"{metrics['Ppk']:.4f}")

    # 💡 [버그 조치 완전 정제 완료]: 지저분한 레퍼런스 표를 삭제하고 시인성이 극대화된 처방 카드로 개편
    cp_score = metrics['Cpk']
    if cp_score >= 1.67:
        grade, status, action = "최우수 (0등급)", "공정능력이 매우 충분함", "들쭉날쭉이 약간 커져도 걱정할 필요가 없다. 비용절감이나 관리의 간소화를 생각하도록 한다."
    elif cp_score >= 1.33:
        grade, status, action = "우수 (1등급)", "공정능력 충분함", "아주 이상적인 공정상황이므로 현재의 상태를 유지한다."
    elif cp_score >= 1.00:
        grade, status, action = "보통 (2등급)", "공정능력이 충분하지는 않지만 그 정도면 괜찮다", "공정관리를 확실하게 하여 관리상태를 유지할 것. Cp가 1에 가까워지면 불량발생의 가능성이 있으므로 주의해야 한다."
    elif cp_score >= 0.67:
        grade, status, action = "부족 (3등급)", "공정능력이 모자란다", "불량품이 생기고 있다. 전체 선별, 공정의 개선, 관리가 필요하다."
    else:
        grade, status, action = "불량 (4등급)", "공정능력 매우 부족하다", "품질이 전혀 만족스럽지 않다. 서둘러 현황조사, 원인규명, 품질개선 같은 긴급 대책을 펴야 한다. 상한 하한 규격 값의 재검토도 해야 한다."

    st.success(f"**🏅 공정 품질 등급 판정 결과: {grade} ({status})**\n\n👉 **종합 처방 조치 권고사항:** {action}")
    
    # 확장된 정규분포 연속 PDF 곡선 오버레이 시각화 대시보드 출력
    fig_hist = vis.plot_process_capability_histogram(final_df, val_col, final_lsl, final_usl, metrics)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("📉 4단계: Shewhart 제어 모니터링 관리도 (Statistical Process Control)")
    chart_mode = st.selectbox("가동 모니터링 관리도 유형 커스텀 선택", ["Xbar-R", "Xbar-s", "I-MR"])

    window_param = 3
    if chart_mode == "I-MR":
        window_param = st.slider("Individual Moving Range 이동 윈도우 크기(w) 설정", min_value=2, max_value=10, value=3)

    chart1, chart2 = engine.generate_value_chart_data(df_raw, sg_col, val_col, chart_type=chart_mode, window=window_param)
    ooc_points = chart1[(chart1['point'] > chart1['UCL']) | (chart1['point'] < chart1['LCL'])].index.tolist()

    if ooc_points:
        st.error(f"🚨 [이상 원인 탐지 알림]: 관리 한계(Control Limits)를 이탈한 부적합 부분군 노드 발견: {ooc_points}")
        exclude_ooc = st.checkbox("이상 부분군 제거 후 관리도 재작성 가동 (강의록 23페이지 구현)")
        if exclude_ooc:
            cleaned_df = df_raw[~df_raw[sg_col].isin(ooc_points)].copy()
            chart1, chart2 = engine.generate_value_chart_data(cleaned_df, sg_col, val_col, chart_type=chart_mode, window=window_param)
            st.success(f"이상 부분군 제거 완료: {len(df_raw)}개 행 -> {len(cleaned_df)}개 행으로 재작성 가동")
    else:
        st.success("🎯 모든 계측 데이터가 관리 한계선 내부에서 안정적으로 주행 중입니다.")

    fig_control = vis.plot_control_chart_dashboard(chart1, chart2, chart_type=chart_mode)
    st.plotly_chart(fig_control, use_container_width=True)
