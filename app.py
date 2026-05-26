import streamlit as st
import pandas as pd
import numpy as np
import stats_engine as engine
import visualizer as vis

st.set_page_config(layout="wide", page_title="스마트제조 공정품질 분석 시스템")

st.title("🏭 스마트제조 공정능력분석 & 통계적 공정관리(SPC) 웹 대시보드")
st.write("강의록 표준 설계 알고리즘 기반 데이터 통합 분석 및 실시간 리포팅 시스템")

# --- 1. 사이드바: 데이터 제어 및 규격 설정 패널 ---
st.sidebar.header("📁 데이터 분석 조건 제어반")

# 실습 가용 기본 데이터 빌더 (파일 없을 시 폴백 구동용)
@st.cache_data
def load_default_sample_data():
    np.random.seed(42)
    lines = [f"pl_{i}" for i in range(1, 7)]
    rows = []
    for line in lines:
        shift = np.random.uniform(-10, 10)
        samples = np.random.normal(loc=3500 + shift, scale=120, size=5)
        for val in samples:
            rows.append({"prod_line": line, "viscocity": val})
    return pd.DataFrame(rows)

uploaded_file = st.sidebar.file_uploader("CSV 혹은 엑셀 파일 업로드", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
else:
    df_raw = load_default_sample_data()
    st.info("⚠️ 업로드된 파일이 없어 강의록 9페이지 기반의 PVC 점도 샘플 데이터셋을 로드했습니다.")

# 컬럼 선택 상자 동적 배치
all_cols = df_raw.columns.tolist()
sg_col = st.sidebar.selectbox("부분군(Subgroup) 식별 구분 컬럼 선택", all_cols, index=0)
val_col = st.sidebar.selectbox("계측 데이터 특성치(Value) 컬럼 선택", all_cols, index=1 if len(all_cols) > 1 else 0)

# 가이드 목표/규격 경계치 입력 인터페이스
default_mean = float(df_raw[val_col].mean())
st.sidebar.subheader("🎯 공정 제어 한계 규격(Specification)")
target_val = st.sidebar.number_input("고객 품질 목표값(Target)", value=round(default_mean, 2))
tolerance = st.sidebar.number_input("허용 공차 한계폭(Tolerance)", value=500.0)

lsl = target_val - tolerance
usl = target_val + tolerance

st.sidebar.markdown(f"**확정 규격 상/하한값**")
st.sidebar.info(f"🔴 **USL**: {usl:.2f} \n\n🔵 **LSL**: {lsl:.2f}")

# --- 2. 메인 데이터 검토 화면 뷰어 ---
st.header("🔍 1단계: 실시간 수집 공정 데이터 구조 분석")
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("데이터 데이터프레임 구조 요약")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"전체 가용 관측치 개수: {len(df_raw)}행 | 검측 분석 대상 변수명: {val_col}")

with col2:
    fig_box = vis.plot_boxplot_dashboard(df_raw, sg_col, val_col)
    st.plotly_chart(fig_box, use_container_width=True)

# --- 3. 통계적 정규성 검증 분기점 ---
st.header("📊 2단계: 정규성(Normality Test) 적합 검증")
p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)

c_test1, c_test2 = st.columns([1, 1])
with c_test1:
    st.metric(label="Shapiro-Wilk Test p-value", value=f"{p_value:.6f}")
    if is_normal:
        st.success("✅ 정규성 조건 만족: 원본 데이터 기반으로 즉시 가용 공정능력지수를 연산합니다.")
        final_df = df_raw.copy()
        final_lsl, final_usl = lsl, usl
    else:
        st.warning("⚠️ 정규성 조건 불만족 (p < 0.05): 강의록 지침에 의거하여 Box-Cox 거듭제곱 변환 알고리즘을 강제 구동합니다.")
        
        # 음수 데이터 유무 사전 스캔 처리
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
            st.experimental_rerun if st.button("변환 스케일 화면 갱신 반영") else None

with c_test2:
    fig_qq = vis.plot_qq_dashboard(final_df[val_col].values)
    st.plotly_chart(fig_qq, use_container_width=True)

# --- 4. 공정능력지수 연산 리포팅 ---
st.header("📈 3단계: 단기/장기 잠재 공정능력(Process Capability Analysis) 평가")
metrics = engine.analyze_process_capability(final_df, sg_col, val_col, final_lsl, final_usl)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("단기 잠재력 (산포만 반영) Cp", f"{metrics['Cp']:.4f}")
m_col2.metric("단기 실효 능력 (치우침 고려) Cpk", f"{metrics['Cpk']:.4f}")
m_col3.metric("장기 잠재력 (종합 변동) Pp", f"{metrics['Pp']:.4f}")
m_col4.metric("장기 실효 능력 (종합 실효) Ppk", f"{metrics['Ppk']:.4f}")

# 등급 및 조치 판정 가이드 출력 테이블 (강의록 9페이지 100% 매핑 보존)
cp_score = metrics['Cpk']
if cp_score >= 1.67:
    grade, status, action = "최우수 (0등급)", "공정능력이 매우 충분함", "관리 간소화 및 공정 비용 절감 방안 검토 권장"
elif cp_score >= 1.33:
    grade, status, action = "우수 (1등급)", "공정능력 충분함", "상당히 이상적인 상태이므로 현재 공정 가동 조건 유지"
elif cp_score >= 1.00:
    grade, status, action = "보통 (2등급)", "공정능력 괜찮은 수준이나 근접 이탈 주의", "공정 관리를 확실하게 모니터링하여 불량 예방 주의"
elif cp_score >= 0.67:
    grade, status, action = "부족 (3등급)", "공정능력 모자람 (불량 발생 중)", "전수 검사 및 공정 설비 파라미터 개선 개선 조치 긴급 처방 필요"
else:
    grade, status, action = "불량 (4등급)", "공정능력 매우 부족함", "품질 전면 부적합 수준. 서둘러 현황조사 원인 규명 및 규격값 재검토"

st.help({"품질 등급 분류 판정 결과": grade, "공정 상태 평가 정보": status, "종합 처방 조치 권고사항": action})

fig_hist = vis.plot_process_capability_histogram(final_df, val_col, final_lsl, final_usl, metrics)
st.plotly_chart(fig_hist, use_container_width=True)

# --- 5. 통계적 공정관리(SPC) 제어 차트 대시보드 ---
st.header("📉 4단계: Shewhart 제어 모니터링 관리도 (Statistical Process Control)")

chart_mode = st.selectbox("가동 모니터링 관리도 유형 커스텀 선택", ["Xbar-R", "Xbar-s", "I-MR"])

# 개별 이동범위 윈도우 조절바 (I-MR용)
window_param = 3
if chart_mode == "I-MR":
    window_param = st.slider("Individual Moving Range 이동 윈도우 크기(w) 설정", min_value=2, max_value=10, value=3)

chart1, chart2 = engine.generate_value_chart_data(df_raw, sg_col, val_col, chart_type=chart_mode, window=window_param)

# 관리 한계선(UCL/LCL) 이탈 여부 탐지 및 가이드 스캔 로직 (강의록 22~23페이지 프로세스 보존)
ooc_points = chart1[(chart1['point'] > chart1['UCL']) | (chart1['point'] < chart1['LCL'])].index.tolist()

if ooc_points:
    st.error(f"🚨 [이상 원인 탐지 알림]: 관리 한계(Control Limits)를 이탈한 부적합 부분군 노드가 발견되었습니다: {ooc_points}")
    st.markdown("💡 **조치 안내:** 이상 원인이 확실한 공정 에러 데이터인 경우, 아래 옵션을 체크하여 해당 로트를 정제 후 경계 한계선을 재설정하여 운영하십시오.")
    exclude_ooc = st.checkbox("이상치 부분군 자동 정제 후 관리도 재작성 시뮬레이션 가동")
    
    if exclude_ooc:
        cleaned_df = df_raw[~df_raw[sg_col].isin(ooc_points)].copy()
        chart1, chart2 = engine.generate_value_chart_data(cleaned_df, sg_col, val_col, chart_type=chart_mode, window=window_param)
        st.success(f"정제 처리 완료: 이상 데이터 {len(df_raw)-len(cleaned_df)}건을 제거한 데이터로 관리한계를 재구성했습니다.")
else:
    st.success("🎯 현재 모든 계측 노드가 관리선(Control Limits) 한계 내부에 정착한 이상 원인 없는 안정 상태 공정입니다.")

fig_control = vis.plot_control_chart_dashboard(chart1, chart2, chart_type=chart_mode)
st.plotly_chart(fig_control, use_container_width=True)