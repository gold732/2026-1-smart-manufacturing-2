import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import numpy as np

def plot_boxplot_dashboard(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, color=x_col, points='all', title=f'[{y_col}] 부품/라인별 박스플롯 분포 분석', template='seaborn')
    return fig

def plot_qq_dashboard(values):
    z_value = stats.zscore(values)
    (x, y), _ = stats.probplot(z_value, dist='norm')
    fig = px.scatter(x=x, y=y, title='정규성 검증을 위한 Q-Q Plot', labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, template='seaborn')
    fig.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
    fig.update_layout(width=400, height=400)
    return fig

def plot_process_capability_histogram(df, val_col, lsl, usl, metrics):
    """강의록의 핵심 사양인 히스토그램 위 정규분포 PDF 오버레이 곡선 추가 구현"""
    fig = px.histogram(df, x=val_col, nbins=15, title=f'{val_col.capitalize()} 공정능력 히스토그램 및 확률밀도 분석', opacity=0.5, template='seaborn')
    
    # 오버레이용 연속 확률 분포 곡선선 생성
    x_min, x_max = df[val_col].min(), df[val_col].max()
    x_axis = np.linspace(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1, 200)
    
    # 히스토그램 빈 간격 및 데이터 수 비례 캘리브레이션
    counts, bins = np.histogram(df[val_col], bins=15)
    scale_factor = len(df) * (bins[1] - bins[0])
    
    # 단기 및 장기 곡선 산출 연산
    y_pdf_within = stats.norm.pdf(x_axis, loc=metrics['x_bar'], scale=metrics['sigma_within']) * scale_factor
    y_pdf_overall = stats.norm.pdf(x_axis, loc=metrics['x_bar'], scale=metrics['sigma_overall']) * scale_factor
    
    fig.add_trace(go.Scatter(x=x_axis, y=y_pdf_within, mode='lines', line=dict(color='darkblue', width=2), name='이론 단기거동(Within)'))
    fig.add_trace(go.Scatter(x=x_axis, y=y_pdf_overall, mode='lines', line=dict(color='crimson', width=1.5, dash='dash'), name='이론 장기거동(Overall)'))
    
    fig.add_vline(x=lsl, line_width=2, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_width=2, line_dash="dash", line_color="red", annotation_text="USL")
    fig.update_layout(barmode='overlay')
    return fig

def plot_control_chart_dashboard(chart1, chart2, chart_type='Xbar-R'):
    names = chart_type.split('-')
    sub_title1, sub_title2 = names[0], names[1] if len(names) > 1 else 'MR'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'{sub_title1} 관리도', f'{sub_title2} 관리도'))
    
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['point'], mode='lines+markers', name=sub_title1, marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['point'], mode='lines+markers', name=sub_title2, marker=dict(size=6, color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=2, col=1)
    
    for i, c_data in enumerate([chart1, chart2], start=1):
        last_idx = c_data.index[-1]
        fig.add_annotation(x=last_idx, y=c_data['UCL'].iloc[-1], text=f"UCL={c_data['UCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='magenta'), row=i, col=1, xshift=35)
        fig.add_annotation(x=last_idx, y=c_data['CL'].iloc[-1], text=f"CL={c_data['CL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='green'), row=i, col=1, xshift=35)
        fig.add_annotation(x=last_idx, y=c_data['LCL'].iloc[-1], text=f"LCL={c_data['LCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='red'), row=i, col=1, xshift=35)

    fig.update_layout(height=650, title_text=f"{chart_type} 통계적 모니터링 관리 시각화 대시보드", showlegend=False, template='seaborn')
    return fig
