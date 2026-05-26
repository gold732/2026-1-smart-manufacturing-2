import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import stats_engine as engine

def render(df_raw, sg_col, val_col, lsl, usl, sigma_method):
    st.header("📈 3단계: 잠재 및 실효 공정능력지수(Process Capability) 평가")
    st.write("공정의 장/단기 변동을 정량화하고 품질 만족 한계 상태를 진단합니다.")
    
    # 정규성 검정에 따른 동적 데이터 변환 분기 처리
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    if is_normal:
        final_df, final_lsl, final_usl = df_raw.copy(), lsl, usl
    else:
        if (df_raw[val_col] <= 0).any():
            st.error("❌ Box-Cox 변환 오류: 데이터에 0 또는 음수가 있어 원본 스케일로 연산합니다.")
            final_df, final_lsl, final_usl = df_raw.copy(), lsl, usl
        else:
            transformed, lsl_t, usl_t, lambda_val = engine.apply_box_cox(df_raw[val_col].values, lsl, usl)
            final_df = df_raw.copy()
            final_df[val_col] = transformed
            final_lsl, final_usl = lsl_t, usl_t
            st.info(f"💡 정규화 완료: Box-Cox 최적 Lambda (λ) = {lambda_val:.4f}를 적용하여 수식을 연산합니다.")

    # 사이드바에서 전달받은 한글 라디오 버튼 매핑 매개변수 적용
    engine_method = "Pooled Standard Deviation" if "합동" in sigma_method else "Subgroup Range Average"
    metrics = engine.analyze_process_capability(final_df, sg_col, val_col, final_lsl, final_usl, method=engine_method)
    
    # 정량 스코어보드 출력
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("단기 잠재 능력 Cp (산포만 고려)", f"{metrics['Cp']:.4f}")
    score_col2.metric("단기 실효 능력 Cpk (치우침 고려)", f"{metrics['Cpk']:.4f}")
    score_col3.metric("장기 잠재 능력 Pp (종합산포 고려)", f"{metrics['Pp']:.4f}")
    score_col4.metric("장기 실효 능력 Ppk (종합실효 고려)", f"{metrics['Ppk']:.4f}")
    
    # 등급 가이드 조치 카드 리포팅 (강의록 9페이지 기준 완벽 매핑)
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

    st.success(f"**🏅 품질 공정 등급 판정: {grade} ({status})**\n\n👉 **현장 시정 조치 지침:** {action}")
    
    # 확률밀도함수(PDF) 오버레이 히스토그램 시각화
    fig = px.histogram(final_df, x=val_col, nbins=15, title="공정 실측 분포 vs 이론 가우스 곡선 대조분석", opacity=0.4, template='seaborn')
    x_axis = np.linspace(final_df[val_col].min()*0.95, final_df[val_col].max()*1.05, 200)
    counts, bins = np.histogram(final_df[val_col], bins=15)
    scale = len(final_df) * (bins[1] - bins[0])
    
    y_within = stats.norm.pdf(x_axis, loc=metrics['x_bar'], scale=metrics['sigma_within']) * scale
    y_overall = stats.norm.pdf(x_axis, loc=metrics['x_bar'], scale=metrics['sigma_overall']) * scale
    
    fig.add_trace(go.Scatter(x=x_axis, y=y_within, mode='lines', line=dict(color='darkblue', width=2), name='단기이론곡선 (Within)'))
    fig.add_trace(go.Scatter(x=x_axis, y=y_overall, mode='lines', line=dict(color='crimson', width=1.5, dash='dash'), name='장기이론곡선 (Overall)'))
    fig.add_vline(x=final_lsl, line_width=2, line_dash="solid", line_color="blue", annotation_text="LSL")
    fig.add_vline(x=final_usl, line_width=2, line_dash="solid", line_color="red", annotation_text="USL")
    
    fig.update_layout(barmode='overlay', margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)
