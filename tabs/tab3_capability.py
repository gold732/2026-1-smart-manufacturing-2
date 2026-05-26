import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import stats_engine as engine

def render(df_raw, sg_col, val_col):
    st.header("📈 3단계 대시보드: 장단기 잠재 공정능력지수 분석 및 등급 조치")
    
    lsl = st.session_state.get('lsl', 0)
    usl = st.session_state.get('usl', 0)
    sigma_method = st.session_state.get('sigma_method', "합동")
    
    engine_method = "Pooled Standard Deviation" if "합동" in sigma_method else "Subgroup Range Average"
    
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    if is_normal:
        final_df, final_lsl, final_usl = df_raw.copy(), lsl, usl
    else:
        if (df_raw[val_col] <= 0).any():
            final_df, final_lsl, final_usl = df_raw.copy(), lsl, usl
        else:
            transformed, lsl_t, usl_t, lambda_val = engine.apply_box_cox(df_raw[val_col].values, lsl, usl)
            final_df = df_raw.copy()
            final_df[val_col] = transformed
            final_lsl, final_usl = lsl_t, usl_t

    metrics = engine.analyze_process_capability(final_df, sg_col, val_col, final_lsl, final_usl, method=engine_method)
    
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("단기 잠재 능력 Cp", f"{metrics['Cp']:.4f}")
    score_col2.metric("단기 실효 능력 Cpk", f"{metrics['Cpk']:.4f}")
    score_col3.metric("장기 잠재 능력 Pp", f"{metrics['Pp']:.4f}")
    score_col4.metric("장기 실효 능력 Ppk", f"{metrics['Ppk']:.4f}")
    
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

    st.success(f"**🏅 강의록 9p 품질 판정 결과: {grade} ({status})**\n\n👉 **현장 시정 조치 지침:** {action}")
    
    fig = px.histogram(final_df, x=val_col, nbins=15, title="공정 통합 분포 및 확률밀도 오버레이 분석", opacity=0.5, template='seaborn')
    x_axis = np.linspace(final_df[val_col].min()*0.95, final_df[val_col].max()*1.05, 200)
    counts, bins = np.histogram(final_df[val_col], bins=15)
    scale = len(final_df) * (bins[1] - bins[0])
    y_within = stats.norm.pdf(x_axis, loc=metrics['x_bar'], scale=metrics['sigma_within']) * scale
    fig.add_trace(go.Scatter(x=x_axis, y=y_within, mode='lines', line=dict(color='darkblue', width=2), name='단기이론곡선(Within)'))
    fig.add_vline(x=final_lsl, line_width=2, line_dash="solid", line_color="blue", annotation_text="LSL")
    fig.add_vline(x=final_usl, line_width=2, line_dash="solid", line_color="red", annotation_text="USL")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧱 강의록 표준 사양: 부분군별 독립 그리드 히스토그램")
    fig_facet = px.histogram(df_raw, x=val_col, facet_row=sg_col, nbins=15, opacity=0.6, template='seaborn', height=140 * df_raw[sg_col].nunique())
    fig_facet.add_vline(x=lsl, line_width=1.5, line_dash="dash", line_color="red")
    fig_facet.add_vline(x=usl, line_width=1.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig_facet, use_container_width=True)
