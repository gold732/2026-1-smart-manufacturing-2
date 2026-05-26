import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import numpy as np

def plot_boxplot_dashboard(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, color=x_col, points='all', title=f'[{y_col}] 부품/라인별 단기 산포 분포 현황', template='seaborn')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_qq_dashboard(values):
    z_value = stats.zscore(values)
    (x, y), _ = stats.probplot(z_value, dist='norm')
    fig = px.scatter(x=x, y=y, title='정규성 검증 정밀 Q-Q Plot', labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, template='seaborn')
    fig.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
    fig.update_layout(width=380, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_process_capability_histogram(df, val_col, lsl, usl, metrics):
    """강의록 23페이지 형식의 이론적 정규분포 PDF 곡선선이 동적 오버레이되는 히스토그램 구현"""
    # 1. 원본 히스토그램 레이어 구축
    fig = px.histogram(df, x=val_col, nbins=15, title=f'{val_col.capitalize()} 공정 고유 거동 밀도 및 스펙 매칭 분석', opacity=0.4, template='seaborn')
    
    # 2. 정규분포 PDF 이론곡선 연산 및 라인 플로팅 오버레이 추가
    x_min, x_max = df[val_col].min(), df[val_col].max()
    x_range = np.linspace(x_min - (x_max-x_min)*0.2, x_max + (x_max-x_min)*0.2, 200)
    # 히스토그램의 데이터 카운트 스케일에 밀도 스케일 보정 매칭
    counts, bins = np.histogram(df[val_col], bins=15)
    bin_width = bins[1] - bins[0]
    total_count = len(df)
    
    y_pdf = stats.norm.pdf(x_range, loc=metrics['x_bar'], scale=metrics['sigma_within']) * total_count * bin_width
    fig.add_trace(go.Scatter(x=x_range, y=y_pdf, mode='lines', line=dict(color='darkblue', width=2.5), name='단기이론분포(Within)'))
    
    # 장기 변동분포 PDF 곡선선 추가
    y_pdf_overall = stats.norm.pdf(x_range, loc=metrics['x_bar'], scale=metrics['sigma_overall']) * total_count * bin_width
    fig.add_trace(go.Scatter(x=x_range, y=y_pdf_overall, mode='lines', line=dict(color='crimson', width=1.5, dash='dash'), name='장기종합분포(Overall)'))

    # 3. 인프라 규격 가이드선 적재
    fig.add_vline(x=lsl, line_width=2.5, line_dash="solid", line_color="blue", annotation_text="LSL")
    fig.add_vline(x=usl, line_width=2.5, line_dash="solid", line_color="red", annotation_text="USL")
    fig.add_vline(x=metrics['x_bar'], line_width=1.5, line_dash="dot", line_color="green", annotation_text="Target(X-Bar)")
    
    fig.update_layout(height=450, margin=dict(l=30, r=30, t=50, b=30), barmode='overlay')
    return fig

def plot_control_chart_dashboard(chart1, chart2, chart_type='Xbar-R'):
    """강의록 09_통계적공정관리 12~13페이지 선조합 제어 모니터링 대시보드 완벽 재현"""
    names = chart_type.split('-')
    sub_title1, sub_title2 = names[0], names[1] if len(names) > 1 else 'MR'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'● {sub_title1} 품질 변동 중심 제어 관리도', f'● {sub_title2} 공정 산포 추적 관리도'))
    
    # 상단 메인 계측점
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['point'], mode='lines+markers', name=sub_title1, marker=dict(size=8, color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['CL'], mode='lines', line=dict(color='green', dash='dashdot', width=1.5), name='CL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['LCL'], mode='lines', line=dict(color='red', dash='dot', width=1.5), name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['UCL'], mode='lines', line=dict(color='magenta', dash='dot', width=1.5), name='UCL'), row=1, col=1)
    
    # 하단 산포 폭
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['point'], mode='lines+markers', name=sub_title2, marker=dict(size=8, color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['CL'], mode='lines', line=dict(color='green', dash='dashdot', width=1.5), name='CL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['LCL'], mode='lines', line=dict(color='red', dash='dot', width=1.5), name='LCL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['UCL'], mode='lines', line=dict(color='magenta', dash='dot', width=1.5), name='UCL'), row=2, col=1)
    
    # 우측 마진 고정 주석(Annotation) 수치 표시 최적화
    for idx, chart_df in enumerate([chart1, chart2], start=1):
        last_node = chart_df.index[-1]
        ucl_val, cl_val, lcl_val = chart_df['UCL'].iloc[-1], chart_df['CL'].iloc[-1], chart_df['LCL'].iloc[-1]
        
        fig.add_annotation(x=last_node, y=ucl_val, text=f"UCL={ucl_val:.2f}", showarrow=False, font=dict(color='magenta', size=11), row=idx, col=1, xshift=45)
        fig.add_annotation(x=last_node, y=cl_val, text=f"CL={cl_val:.2f}", showarrow=False, font=dict(color='green', size=11), row=idx, col=1, xshift=45)
        fig.add_annotation(x=last_node, y=lcl_val, text=f"LCL={lcl_val:.2f}", showarrow=False, font=dict(color='red', size=11), row=idx, col=1, xshift=45)

    fig.update_layout(height=600, showlegend=False, template='seaborn', margin=dict(l=40, r=80, t=60, b=40))
    return fig
