import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import stats_engine as engine

def render(df_raw, sg_col, val_col):
    st.header("📈 장단기 공정능력지수 분석 및 진단")
    st.write("공정의 단기 잠재 능력과 장기 성능 변동을 정량화하고 품질 만족 한계를 진단합니다.")
    
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
        grade, status, action = "최우수 (0등급)", "공정능력이 매우 충분함", "변동이 약간 커져도 안전한 수준입니다. 관리의 간소화 및 공정 최적화를 검토하십시오."
    elif cp_score >= 1.33:
        grade, status, action = "우수 (1등급)", "공정능력 충분함", "이상적인 공정 상황이므로 현재 가동 상태를 유지하십시오."
    elif cp_score >= 1.00:
        grade, status, action = "보통 (2등급)", "공정능력이 충분하지는 않으나 가용 수준", "확실한 공정 관리가 필요합니다. 지수가 1에 가까워지면 불량이 발생할 수 있으므로 주의하십시오."
    elif cp_score >= 0.67:
        grade, status, action = "부족 (3등급)", "공정능력이 모자람", "현재 불량이 발생하고 있습니다. 공정 개선 및 원인 분석이 시급합니다."
    else:
        grade, status, action = "불량 (4등급)", "공정능력 매우 부족함", "품질이 만족스럽지 못하므로 현황 조사, 원인 규명 및 규격의 재검토를 권장합니다."

    st.success(f"**🏅 공정 품질 등급 판정 결과: {grade} ({status})**\n\n👉 **현장 종합 조치 권고사항:** {action}")
    
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

    st.subheader("🧱 부분군별 독립 분포 비교 그리드")
    fig_facet = px.histogram(df_raw, x=val_col, facet_row=sg_col, nbins=15, opacity=0.6, template='seaborn', height=140 * df_raw[sg_col].nunique())
    fig_facet.add_vline(x=lsl, line_width=1.5, line_dash="dash", line_color="red")
    fig_facet.add_vline(x=usl, line_width=1.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig_facet, use_container_width=True)
