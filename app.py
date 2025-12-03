#!/usr/bin/env python
# coding: utf-8


from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import dash
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from functools import lru_cache

# ------------------------------------------------------------
# 1) OPTIMIZED DATA LOADING WITH CACHING
# ------------------------------------------------------------

class DataManager:
    """Optimized data manager with lazy loading and caching"""
    _main = None
    _geo = None
    _calls = None
    
    @classmethod
    def get_main(cls):
        if cls._main is None:
            cls._main = pd.read_parquet("data/main_clean.parquet")
            cls._main["Created Month"] = cls._main["Created Time"].dt.to_period("M").astype(str)
            cls._main["is_real_paid"] = (
                (cls._main["Initial Amount Paid"] > 9) & 
                (cls._main["is_trial"] == 0) & 
                (cls._main["is_demo_payment"] == 0)
            )
        return cls._main
    
    @classmethod
    def get_calls(cls):
        if cls._calls is None:
            cls._calls = pd.read_parquet("data/Project_Calls.parquet")
            cls._calls["Call Start Time"] = pd.to_datetime(cls._calls["Call Start Time"], errors="coerce")
            cls._calls["call_month"] = cls._calls["Call Start Time"].dt.to_period("M").astype(str)
            cls._calls["call_weekday_name"] = cls._calls["Call Start Time"].dt.day_name()
            cls._calls["call_hour"] = cls._calls["Call Start Time"].dt.hour
        return cls._calls

# ------------------------------------------------------------
# 2) THEME (light)
# ------------------------------------------------------------
THEMES = {
    "light": {
        "theme_color": "#f7f9fb",
        "accent_color": "#4e79a7",
        "text_color": "#2c3e50",
        "plot_bg": "#FFFFFF",
        "paper_bg": "#FFFFFF",
        "kpi_bg": "#f7f9fb",
        "axis_line": "#ccd0d5",
        "grid_line": "#e5e7eb",
        "border": "#e1e4e8",
        "widget_bg": "#FFFFFF",
        "widget_text": "#2c3e50",
    }
}
TH = THEMES["light"]

# ------------------------------------------------------------
# 3) KPI COMPONENT
# ------------------------------------------------------------
def kpi_div(title, value, sub=None):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "color": TH["text_color"]}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": TH["text_color"]}),
            html.Div(f"({sub})", style={"fontSize": "13px", "color": "gray", "marginTop": "2px"}) if sub else html.Div()
        ],
        style={
            "flex": 1, "minWidth": "180px",
            "backgroundColor": TH["kpi_bg"], "borderRadius": "10px",
            "padding": "10px", "textAlign": "center",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.15)"
        }
    )

# ------------------------------------------------------------
# 4) OPTIMIZED HELPER FUNCTIONS FOR MARKETING
# ------------------------------------------------------------
def _safe_div(a, b, mul=1.0):
    """Safe division without errors on zero division."""
    if isinstance(a, (pd.Series, np.ndarray)) or isinstance(b, (pd.Series, np.ndarray)):
        a = np.where(pd.isna(a), 0, a)
        b = np.where(pd.isna(b), 0, b)
        result = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        return result * mul
    if pd.isna(a) or pd.isna(b) or b == 0:
        return 0.0
    return (a / b) * mul

@lru_cache(maxsize=32)
def build_layers_no_inflation_cached(df_hash):
    """Cached version of marketing data aggregation."""
    df_f = DataManager.get_main().copy()
    return build_layers_no_inflation(df_f)

def build_layers_no_inflation(df_f: pd.DataFrame):
    """Aggregates spend and sales without campaign inflation."""
    df_f = df_f.copy()
    df_f["Spend"] = pd.to_numeric(df_f["Spend"], errors="coerce").fillna(0)
    df_f["Impressions"] = pd.to_numeric(df_f["Impressions"], errors="coerce").fillna(0)
    df_f["Clicks"] = pd.to_numeric(df_f["Clicks"], errors="coerce").fillna(0)

    spend_level = (
        df_f.groupby(["_month", "Source", "Campaign"], as_index=False)
        .agg(Impressions=("Impressions", "sum"),
             Clicks=("Clicks", "sum"),
             Spend=("Spend", "mean"))
    )

    paid = (
        df_f.query("Stage == 'Payment Done' and `Initial Amount Paid` > 0")
        .sort_values("Id")
        .groupby("Id", as_index=False)
        .agg(Contact=("Contact Name", "first"),
             Product=("Product", "first"),
             EduType=("Education Type", "first"),
             Source=("Source", "first"),
             Campaign=("Campaign", "first"),
             _month=("_month", "first"),
             Paid=("Initial Amount Paid", "sum"))
    )

    unit = (
        paid.groupby(["_month", "Source", "Campaign"], as_index=False)
        .agg(Revenue=("Paid", "sum"),
             B=("Contact", "nunique"),
             T=("Id", "nunique"))
    )

    prod = (
        paid.groupby("Product", as_index=False)
        .agg(B=("Contact", "nunique"),
             T=("Id", "nunique"),
             Revenue=("Paid", "sum"))
    )
    return spend_level, unit, paid, prod

def get_effective_spend(spend_level, paid):
    """Sum of spend only for campaigns with sales."""
    if paid.empty:
        return 0
    keys_paid = paid[['_month', 'Source', 'Campaign']].drop_duplicates()
    spend_effective = spend_level.merge(keys_paid, on=['_month', 'Source', 'Campaign'], how='inner')
    return spend_effective['Spend'].sum()

def compute_unit_kpis(spend_level, paid, df_f, cogs_percent=0.30, cogs_dynamic=True, AC_override=None):
    """Calculates aggregated marketing KPIs and unit economics."""
    
    # Deal preparation
    df_deal_level = (
        df_f.groupby("Id", as_index=False)
        .agg({
            "Stage": "last",
            "Initial Amount Paid": "max", 
            "Offer Total Amount": "max",
            "is_trial": "max",
            "is_demo_payment": "max",
            "Product": "first"
        })
    )

    df_deal_level["Initial Amount Paid"] = df_deal_level["Initial Amount Paid"].fillna(0)
    df_real = df_deal_level.query("is_trial == False and is_demo_payment == False").copy()

    # Main KPIs
    UA = len(df_deal_level)
    B = df_real.query("Stage == 'Payment Done' and `Initial Amount Paid` > 9").shape[0]
    T = B
    Revenue = df_real.query("Stage == 'Payment Done'")["Initial Amount Paid"].sum()

    # Spend
    AC = AC_override if AC_override is not None else spend_level["Spend"].sum()

    # Dynamic COGS
    AOV = _safe_div(Revenue, T)
    if cogs_dynamic:
        cogs_rate = 0.05 if AOV > 1000 else 0.08
        cogs_fix_per_tx = AOV * 0.02
    else:
        cogs_rate = cogs_percent
        cogs_fix_per_tx = 50.0
        
    Total_COGS = Revenue * cogs_rate + cogs_fix_per_tx * T
    COGS_per_tx = _safe_div(Total_COGS, T)
    
    # Derived metrics
    CPA = _safe_div(AC, B)
    APC = _safe_div(T, B)
    C1 = _safe_div(B, UA, 100)
    CLTV = (AOV - COGS_per_tx) * APC
    LTC = _safe_div(AC, UA)
    CM = Revenue - Total_COGS - AC
    ROI = _safe_div((Revenue - AC), AC, 100)
    GMp = _safe_div((Revenue - Total_COGS), Revenue, 100)

    return dict(
        UA=UA, B=B, T=T, Revenue=Revenue, AC=AC,
        Total_COGS=Total_COGS, COGS_per_tx=COGS_per_tx,
        AOV=AOV, CPA=CPA, APC=APC, C1=C1,
        CLTV=CLTV, LTC=LTC, CM=CM,
        ROI=ROI, GMp=GMp, cogs_dynamic=cogs_dynamic
    )

def calculate_product_unit_economics(df_f, total_marketing_spend):
    """Product efficiency calculation."""
    try:
        # Real deal preparation
        df_deal_level = (
            df_f.groupby("Id", as_index=False)
            .agg({
                "Stage": "last",
                "Initial Amount Paid": "max",
                "Product": "first",
                "Contact Name": "first",
                "Source": "first",
                "Campaign": "first",
                "is_trial": "max",
                "is_demo_payment": "max",
            })
        )
        
        df_real = df_deal_level.query("is_trial == False and is_demo_payment == False").copy()
        real_paid = df_real.query("Stage == 'Payment Done' and `Initial Amount Paid` > 9")
        
        if real_paid.empty:
            return None
        
        # Product sales aggregation
        product_sales = (
            real_paid.groupby('Product', observed=False)
            .agg(
                Customers=('Contact Name', 'nunique'),
                Transactions=('Id', 'count'),
                Revenue=('Initial Amount Paid', 'sum'),
                Avg_Transaction_Value=('Initial Amount Paid', 'mean')
            )
            .reset_index()
        )
        
        # Campaign spend
        product_campaigns = (
            real_paid.groupby(['Product', 'Source', 'Campaign'])
            .agg(Product_Revenue=('Initial Amount Paid', 'sum'))
            .reset_index()
        )
        campaign_spend = (
            df_f.groupby(['Source', 'Campaign'])
            .agg(Spend=('Spend', 'first'))
            .reset_index()
        )
        
        product_spend = (
            product_campaigns.merge(campaign_spend, on=['Source', 'Campaign'], how='left')
            .groupby('Product', observed=False)
            .agg(Product_Spend=('Spend', 'sum'))
            .reset_index()
        )
        
        # Combine sales and spend
        product_metrics = product_sales.merge(product_spend, on='Product', how='left')
        product_metrics['Product_Spend'] = product_metrics['Product_Spend'].fillna(0)
        
        total_allocated_spend = product_metrics['Product_Spend'].sum()
        if total_allocated_spend == 0:
            total_revenue = product_metrics['Revenue'].sum()
            product_metrics['Marketing_Spend'] = (
                product_metrics['Revenue'] / total_revenue * total_marketing_spend
            )
        else:
            scale_factor = total_marketing_spend / total_allocated_spend
            product_metrics['Marketing_Spend'] = product_metrics['Product_Spend'] * scale_factor
        
        # Efficiency metrics
        product_metrics['CPA'] = _safe_div(product_metrics['Marketing_Spend'], product_metrics['Customers'])
        product_metrics['AOV'] = _safe_div(product_metrics['Revenue'], product_metrics['Transactions'])
        product_metrics['APC'] = _safe_div(product_metrics['Transactions'], product_metrics['Customers'])
        product_metrics['LTV'] = product_metrics['AOV'] * product_metrics['APC'] * 1.2
        product_metrics['LTV_CAC_Ratio'] = _safe_div(product_metrics['LTV'], product_metrics['CPA'])
        
        # Cost and profit
        product_metrics['COGS_Rate'] = np.where(
            product_metrics['Avg_Transaction_Value'] > 2000, 0.15,
            np.where(product_metrics['Avg_Transaction_Value'] > 1000, 0.25, 0.35)
        )
        product_metrics['Total_COGS'] = product_metrics['Revenue'] * product_metrics['COGS_Rate']
        product_metrics['Gross_Margin'] = product_metrics['Revenue'] - product_metrics['Total_COGS']
        product_metrics['Contribution_Margin'] = product_metrics['Gross_Margin'] - product_metrics['Marketing_Spend']
        product_metrics['ROI'] = _safe_div(product_metrics['Contribution_Margin'], product_metrics['Marketing_Spend'], 100)
        
        # Efficiency categorization
        product_metrics['Efficiency_Score'] = np.where(
            product_metrics['LTV_CAC_Ratio'] > 2, 'High',
            np.where(product_metrics['LTV_CAC_Ratio'] > 1, 'Medium', 'Low')
        )
        return product_metrics.sort_values('ROI', ascending=False)
    
    except Exception as e:
        print(f"Error in product analytics: {e}")
        return None

# ------------------------------------------------------------
# OPTIMIZED GEO DATA PREPARATION
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def prepare_geo_data_optimized():
    """Optimized geo data preparation with caching"""
    df = pd.read_parquet("data/deals_geo_enriched_full.parquet")
    
    # Select only needed columns
    needed_cols = [
        'City', 'country_iso2', 'lat', 'lng', 'deal_count', 'success_count',
        'Created Month', 'Deutsch_Level_Clean', 'Source_top', 'avg_amount'
    ]
    df = df[needed_cols].copy()
    
    # Country mapping
    country_mapping = {
        'DE': 'Germany', 'AT': 'Austria', 'SK': 'Slovakia', 'PL': 'Poland', 
        'ME': 'Montenegro', 'RS': 'Serbia', 'NL': 'Netherlands', 'AE': 'UAE',
        'HU': 'Hungary', 'US': 'USA', 'CZ': 'Czech Republic', 'FR': 'France', 
        'LV': 'Latvia', 'GB': 'UK', 'IL': 'Israel', 'TH': 'Thailand',
        'BE': 'Belgium', 'MD': 'Moldova', 'RU': 'Russia', 'UA': 'Ukraine',
        'BY': 'Belarus', 'KZ': 'Kazakhstan', 'RO': 'Romania', 'BG': 'Bulgaria',
        'IT': 'Italy', 'ES': 'Spain', 'CH': 'Switzerland', 'DK': 'Denmark',
        'SE': 'Sweden', 'NO': 'Norway', 'FI': 'Finland'
    }
    
    df['country'] = df['country_iso2'].map(country_mapping).fillna('Other countries')
    
    # Optimize data types
    df['deal_count'] = df['deal_count'].astype('int32')
    df['success_count'] = df['success_count'].astype('int32')
    df['lat'] = df['lat'].astype('float32')
    df['lng'] = df['lng'].astype('float32')
    
    # Calculate revenue
    df['revenue'] = df['avg_amount'] * df['success_count']
    df['success_rate'] = (df['success_count'] / df['deal_count'] * 100).fillna(0)
    
    return df

# ------------------------------------------------------------
# DEALS DASHBOARD (OPTIMIZED + ENGLISH)
# ------------------------------------------------------------

def layout_deals():
    main_data = DataManager.get_main()
    
    managers = ["All"] + sorted(main_data["Deal Owner Name"].dropna().unique().tolist())
    stages = ["All"] + sorted(main_data["Stage"].dropna().unique().tolist())
    sources = ["All"] + sorted(main_data["Source"].dropna().unique().tolist())
    campaigns = ["All"] + sorted(main_data["Campaign"].dropna().unique().tolist())
    months = sorted(main_data["Created Month"].dropna().unique().tolist())

    sidebar = html.Div(
        [
            html.H4("Filters", style={"marginTop": 0, "marginBottom": "10px"}),

            html.Label("Manager:", style={"fontWeight": "600", "marginTop": "6px"}),
            dcc.Dropdown(managers, "All", id="deal-manager", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Stage:", style={"fontWeight": "600"}),
            dcc.Dropdown(stages, "All", id="deal-stage", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Source:", style={"fontWeight": "600"}),
            dcc.Dropdown(sources, "All", id="deal-source", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Campaign:", style={"fontWeight": "600"}),
            dcc.Dropdown(campaigns, "All", id="deal-campaign", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Funnel Type:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "Funnel (classic)", "value": "Funnel"},
                    {"label": "Funnelarea", "value": "Funnelarea"},
                    {"label": "Pie", "value": "Pie"},
                ],
                value="Funnel",
                id="deal-funnel-type",
                clearable=False,
                style={"marginBottom": "15px"},
            ),

            html.Label("Period:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Div([
                html.Div([
                    html.Label("From:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[0], id="deal-period-start", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block", "marginRight": "10px"}),

                html.Div([
                    html.Label("To:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[-1], id="deal-period-end", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block"})
            ], style={"marginBottom": "20px"}),

            html.Hr(),
            html.H4("Settings", style={"marginTop": "10px"}),

            html.Label("Metric:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "Win Rate", "value": "win_rate"},
                    {"label": "Average Deal Amount (€)", "value": "avg_amount"},
                ],
                value="win_rate",
                id="deal-metric",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Sort:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "Ascending", "value": True},
                    {"label": "Descending", "value": False},
                ],
                value=False,
                id="deal-sort",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Width:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[1000, 1200, 1300, 1500, 2000, 2500],
                value=1500,
                id="deal-width",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Height:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[600, 700, 800, 900, 1000],
                value=1000,
                id="deal-height",
                clearable=False,
                style={"marginBottom": "15px"},
            ),

            html.Hr(),
            html.H4("Dashboard Navigation", style={"marginTop": "10px", "marginBottom": "8px"}),
            get_switch_buttons("deals"),
        ],
        style={
            "width": "320px",
            "padding": "16px",
            "backgroundColor": TH["theme_color"],
            "borderLeft": f"1px solid {TH['border']}",
            "position": "fixed",
            "right": 0,
            "top": 0,
            "height": "100vh",
            "overflowY": "auto",
            "color": TH["text_color"],
        },
    )

    content = html.Div(
        [
            html.Div(id="deal-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
            dcc.Loading(
                id="loading-deals",
                type="circle",
                children=dcc.Graph(id="deal-figure", config={"displaylogo": False})
            )
        ],
        style={"marginRight": "340px", "padding": "16px", "backgroundColor": TH["paper_bg"], "color": TH["text_color"]}
    )

    return html.Div([content, sidebar], id="deals-layout")

def compute_deals_view(manager, stage, source, campaign,
                       period_start, period_end,
                       metric, sort_asc, w, h, funnel_kind):

    df = DataManager.get_main().copy()

    # Filters
    df = df[(df["Created Month"] >= period_start) & (df["Created Month"] <= period_end)]
    if manager != "All": df = df[df["Deal Owner Name"] == manager]
    if stage != "All": df = df[df["Stage"] == stage]
    if source != "All": df = df[df["Source"] == source]
    if campaign != "All": df = df[df["Campaign"] == campaign]

    # Deal level aggregation
    gr = (df.groupby("Id", as_index=False)
          .agg({
              "Deal Owner Name": "first", "Stage": "last", "Source": "first",
              "Initial Amount Paid": "max", "Offer Total Amount": "max",
              "is_trial": "max", "is_demo_payment": "max", "has_payment": "max",
              "closing_type": "first", "SLA": "max",
              "Created Time": "min", "Closing Date": "max"
          }))
    gr["Initial Amount Paid"] = gr["Initial Amount Paid"].fillna(0)
    gr["Offer Total Amount"] = gr["Offer Total Amount"].fillna(0)

    gr["is_real_payment"] = gr["Initial Amount Paid"] > 9
    gr["is_demo"] = gr["Initial Amount Paid"].between(0, 9, inclusive="both")
    gr["is_won"] = (gr["Stage"].eq("Payment Done")) & (gr["is_real_payment"]) & (~gr["is_trial"])
    gr["is_lost"] = gr["Stage"].eq("Lost")
    gr["is_active"] = ~gr["Stage"].isin(["Lost", "Payment Done"])

    df_real = gr.query("is_trial == False and is_demo_payment == False").copy()

    # KPI
    stage_base = gr.copy()
    stage_base["Stage"] = stage_base["Stage"].fillna("Unknown")

    total_leads = len(stage_base)
    real_paid = stage_base.query("Stage=='Payment Done' and `Initial Amount Paid`>9")
    formal_paid = stage_base.query("Stage=='Payment Done' and `Initial Amount Paid`<=9")
    lost = stage_base.query("Stage=='Lost'")
    active = stage_base.query("Stage not in ['Lost','Payment Done']")

    kpis = [
        kpi_div("Total Leads", f"{total_leads:,}"),
        kpi_div("In Progress", f"{(len(active) / total_leads * 100 if total_leads else 0):.1f}%", f"{len(active):,}"),
        kpi_div("Lost", f"{(len(lost) / total_leads * 100 if total_leads else 0):.1f}%", f"{len(lost):,}"),
        kpi_div("Conversion (real)", f"{(len(real_paid) / total_leads * 100 if total_leads else 0):.1f}%", f"{len(real_paid):,}"),
        kpi_div("Conversion (Trial/Demo)", f"{(len(formal_paid) / total_leads * 100 if total_leads else 0):.1f}%", f"{len(formal_paid):,}"),
        kpi_div("Avg Payment", f"{(real_paid['Initial Amount Paid'].mean() if not real_paid.empty else 0):,.0f} €"),
        kpi_div("Total Revenue", f"{(real_paid['Initial Amount Paid'].sum() if not real_paid.empty else 0):,.0f} €"),
    ]

    # Figure with 4 charts
    funnel_spec = {"type": "domain"} if funnel_kind in ["Pie", "Funnelarea"] else {"type": "xy"}
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, funnel_spec],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            "Deal Dynamics (by Created Time)",
            "Conversion Funnel",
            "Deal Sources (TOP 15)"
        ],
        horizontal_spacing=0.12, vertical_spacing=0.25
    )

    # 1) Dynamics
    if not df_real.empty:
        df_real["Created Month"] = df_real["Created Time"].dt.to_period("M").astype(str)
    deals_month = (df_real.groupby("Created Month", observed=False).size()
                   .reset_index(name="count").sort_values("Created Month")) if not df_real.empty else pd.DataFrame({"Created Month": [], "count": []})
    fig.add_trace(
        go.Bar(x=deals_month["Created Month"], y=deals_month["count"], marker_color=TH["accent_color"], name="Deals Created"),
        row=1, col=1
    )

    # 2) Funnel
    awareness = total_leads
    considering = gr.query("~Stage.isin(['Lost','Payment Done'])").shape[0]
    lost_cnt = gr.query("Stage=='Lost'").shape[0]
    converted = gr.query("(Stage=='Payment Done') or (`Initial Amount Paid`>0) or (is_trial==True) or (is_demo_payment==True)").shape[0]

    fd = pd.DataFrame(
        [("Awareness", awareness),
         ("Consideration", considering),
         ("Lost", lost_cnt),
         ("Conversion", converted)],
        columns=["StageGroup", "count"]
    )
    fd["percent_total"] = (fd["count"] / fd["count"].iloc[0] * 100).round(1)
    fd["percent_step"] = (fd["count"] / fd["count"].shift(1) * 100).round(1)
    fd.iloc[0, fd.columns.get_loc("percent_step")] = 100.0

    if funnel_kind == "Funnelarea":
        fig.add_trace(go.Funnelarea(
            text=fd["StageGroup"], values=fd["percent_total"], textinfo="label+percent",
            marker=dict(colors=px.colors.sequential.Blues_r), name="Funnel"
        ), row=1, col=2)
    elif funnel_kind == "Pie":
        fig.add_trace(go.Pie(
            labels=fd["StageGroup"], values=fd["percent_total"], hole=0.4,
            sort=False, direction="clockwise", rotation=270,
            text=[f"{p:.1f}%" for p in fd["percent_total"]],
            textinfo="label+text", textposition="inside",
            textfont=dict(size=13, color="white"),
            marker=dict(colors=px.colors.sequential.Blues_r),
            name="Funnel"
        ), row=1, col=2)
    else:
        fig.add_trace(go.Funnel(
            y=fd["StageGroup"], x=fd["count"],
            text=[f"{p:.1f}%" for p in fd["percent_total"]],
            textinfo="text+label", textposition="inside",
            textfont=dict(size=14, color="white"),
            marker=dict(color=px.colors.sequential.Blues_r),
            name="Funnel"
        ), row=1, col=2)

    # 3) Sources (TOP 15)
    if "Source" in df_real.columns and not df_real.empty:
        src = (df_real["Source"].value_counts()
               .rename_axis("Source").reset_index(name="count")
               .sort_values("count", ascending=False).head(15))
        fig.add_trace(go.Bar(
            y=src["Source"], x=src["count"], orientation="h",
            marker_color="#59a14f", name="Source"
        ), row=2, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=1)

    # 4) Manager Performance
    perf = DataManager.get_main().copy()
    perf["SLA"] = pd.to_timedelta(perf["SLA"], errors="coerce")
    perf["sla_hours"] = perf["SLA"].dt.total_seconds() / 3600
    perf["is_won"] = (perf["Stage"].eq("Payment Done")) & (perf["Initial Amount Paid"].fillna(0) > 9)
    perf["is_lost"] = perf["Stage"].eq("Lost")

    perf = (perf.groupby("Deal Owner Name", observed=False)
            .agg(total_deals=("Id", "count"),
                 real_wins=("is_won", "sum"),
                 lost=("is_lost", "sum"),
                 avg_sla=("sla_hours", "mean"),
                 avg_amount=("Initial Amount Paid", "mean"))
            .reset_index())
    perf["win_rate"] = (perf["real_wins"] / perf["total_deals"] * 100).round(1)

    if metric == "avg_amount":
        perf = perf.sort_values("avg_amount", ascending=sort_asc)
        y_vals = perf["avg_amount"]
        color_vals = perf["avg_amount"]
        title_text = "Average Deal Amount (€)"
    else:
        perf = perf.sort_values("win_rate", ascending=sort_asc)
        y_vals = perf["win_rate"]
        color_vals = perf["win_rate"]
        title_text = "Win Rate (%)"

    fig.add_trace(go.Bar(
        x=perf["Deal Owner Name"], y=y_vals,
        marker=dict(color=color_vals, colorscale="Blues", showscale=False),
        customdata=perf[["win_rate", "avg_amount", "total_deals", "avg_sla"]],
        hovertemplate=("Manager: %{x}<br>Win rate: %{customdata[0]:.1f}%"
                       "<br>Avg Amount: %{customdata[1]:.0f} €"
                       "<br>Deals: %{customdata[2]}"
                       "<br>Avg SLA: %{customdata[3]:.1f} h<extra></extra>")
    ), row=2, col=2)

    fig.add_annotation(
        text=f"Manager Performance — {title_text}",
        xref="x domain", yref="y domain", x=0.5, y=1.12,
        showarrow=False, font=dict(size=14, color=TH["text_color"], family="Arial"),
        row=2, col=2
    )

    fig.update_layout(
        showlegend=False,
        height=h, width=w,
        plot_bgcolor=TH["plot_bg"], paper_bgcolor=TH["paper_bg"],
        font=dict(color=TH["text_color"]),
        title=dict(text="Deals Performance Dashboard", x=0.5, xanchor="center",
                   font=dict(size=20, color=TH["text_color"]))
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)

    return fig, kpis

# ------------------------------------------------------------
# CALLS DASHBOARD (OPTIMIZED + ENGLISH)
# ------------------------------------------------------------

def layout_calls():
    calls_data = DataManager.get_calls()
    
    managers = ["All"] + sorted(calls_data["Call Owner Name"].dropna().unique().tolist())
    types = ["All"] + sorted(calls_data["Call Type"].dropna().unique().tolist())
    months = sorted(calls_data["call_month"].dropna().unique().tolist())

    sidebar = html.Div(
        [
            html.H4("Filters", style={"marginTop": 0, "marginBottom": "10px"}),

            html.Label("Manager:", style={"fontWeight": "600", "marginTop": "6px"}),
            dcc.Dropdown(managers, "All", id="calls-manager", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Call Type:", style={"fontWeight": "600"}),
            dcc.Dropdown(types, "All", id="calls-type", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Call Filter:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "All Calls", "value": "all"},
                    {"label": "Meaningful (≥3 sec)", "value": "meaningful"},
                    {"label": "Short (<5 sec)", "value": "short"}
                ],
                value="all",
                id="calls-meaning",
                clearable=False,
                style={"marginBottom": "15px"}
            ),

            html.Label("Period:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Div([
                html.Div([
                    html.Label("From:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[0], id="calls-period-start", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block", "marginRight": "10px"}),

                html.Div([
                    html.Label("To:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[-1], id="calls-period-end", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block"})
            ], style={"marginBottom": "20px"}),

            html.Label("Analysis Period:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "By Month", "value": "month"},
                    {"label": "By Weekday", "value": "weekday"}
                ],
                value="month",
                id="calls-time-mode",
                clearable=False,
                style={"marginBottom": "15px"}
            ),

            html.Hr(),
            html.H4("Settings", style={"marginTop": "10px"}),

            html.Label("Sort:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "Ascending", "value": True},
                    {"label": "Descending", "value": False},
                ],
                value=False,
                id="calls-sort",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Width:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[1000, 1200, 1300, 1500, 2000, 2500],
                value=1300,
                id="calls-width",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Height:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[600, 700, 800, 900, 1000],
                value=800,
                id="calls-height",
                clearable=False,
                style={"marginBottom": "15px"},
            ),

            html.Hr(),
            html.H4("Dashboard Navigation", style={"marginTop": "10px", "marginBottom": "8px"}),
            get_switch_buttons("calls"),
        ],
        style={
            "width": "320px",
            "padding": "16px",
            "backgroundColor": TH["theme_color"],
            "borderLeft": f"1px solid {TH['border']}",
            "position": "fixed",
            "right": 0,
            "top": 0,
            "height": "100vh",
            "overflowY": "auto",
            "color": TH["text_color"],
        },
    )

    content = html.Div(
        [
            html.Div(id="calls-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
            dcc.Loading(
                id="loading-calls",
                type="circle",
                children=dcc.Graph(id="calls-figure", config={"displaylogo": False})
            )
        ],
        style={"marginRight": "340px", "padding": "16px", "backgroundColor": TH["paper_bg"], "color": TH["text_color"]}
    )

    return html.Div([content, sidebar], id="calls-layout")

def compute_calls_view(manager, call_type, meaning, period_start, period_end, time_mode, sort_asc, w, h):
    df = DataManager.get_calls().copy()

    # Filters
    df = df[(df["call_month"] >= period_start) & (df["call_month"] <= period_end)]
    if manager != "All": df = df[df["Call Owner Name"] == manager]
    if call_type != "All": df = df[df["Call Type"] == call_type]

    if meaning == "meaningful":
        df = df[df["Call Duration (in seconds)"] >= 3]
    elif meaning == "short":
        df = df[df["Call Duration (in seconds)"] < 5]

    df["is_meaningful_call"] = df["Call Duration (in seconds)"] >= 3

    # KPI
    total_calls = len(df)
    avg_duration = df["Call Duration (in seconds)"].mean() if total_calls else 0
    meaningful_share = (df["is_meaningful_call"].mean() * 100) if total_calls else 0
    short_avg = df.loc[df["Call Duration (in seconds)"] < 5, "Call Duration (in seconds)"].mean() if total_calls else 0
    unique_managers = df["Call Owner Name"].nunique()

    kpis = [
        kpi_div("Total Calls", f"{total_calls:,}"),
        kpi_div("Avg Duration", f"{avg_duration:.1f} sec"),
        kpi_div("Meaningful", f"{meaningful_share:.1f}%", f"{df['is_meaningful_call'].sum():,}"),
        kpi_div("Avg Short Calls", f"{short_avg:.1f} sec"),
        kpi_div("Managers", f"{unique_managers}"),
    ]

    # Charts
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=["Activity by Hour", "Calls Trend", "Call Types"],
        horizontal_spacing=0.12, vertical_spacing=0.25
    )

    # 1) Activity by hour
    hourly = df.groupby("call_hour", observed=False).size().reset_index(name="count")
    fig.add_trace(go.Bar(
        x=hourly["call_hour"], y=hourly["count"],
        marker_color=TH["accent_color"], name="By Hour"
    ), row=1, col=1)

    # 2) Trend
    if time_mode == "month":
        grp = df.groupby("call_month", observed=False).size().reset_index(name="calls")
        fig.add_trace(go.Scatter(
            x=grp["call_month"], y=grp["calls"],
            mode="lines+markers", line_color="#59a14f", name="By Month"
        ), row=1, col=2)
    else:
        grp = (df.groupby("call_weekday_name", observed=False).size()
               .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
               .reset_index(name="calls"))
        fig.add_trace(go.Bar(
            x=grp["call_weekday_name"], y=grp["calls"],
            marker_color="#f28e2b", name="By Day"
        ), row=1, col=2)

    # 3) Call types (Pie)
    types = df["Call Type"].value_counts().rename_axis("Type").reset_index(name="count")
    fig.add_trace(go.Pie(
        labels=types["Type"],
        values=types["count"],
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Blues_r),
        textinfo="label+percent",
        textfont=dict(size=13, color="white")
    ), row=2, col=1)

    # 4) Manager performance
    perf = (df.groupby("Call Owner Name", observed=False)
            .agg(total_calls=("Id", "count"),
                 avg_duration=("Call Duration (in seconds)", "mean"),
                 meaningful_share=("is_meaningful_call", "mean"))
            .reset_index())
    perf["meaningful_share"] = (perf["meaningful_share"] * 100).round(1)
    perf = perf.sort_values("meaningful_share", ascending=sort_asc)

    # Color logic
    if meaning == "meaningful":
        colorscale = "Greens"
    elif meaning == "short":
        colorscale = "Reds"
    else:
        colorscale = "Blues"

    fig.add_trace(go.Bar(
        x=perf["Call Owner Name"],
        y=perf["avg_duration"],
        marker=dict(color=perf["meaningful_share"], colorscale=colorscale, showscale=False),
        customdata=perf[["avg_duration", "meaningful_share"]],
        hovertemplate="Manager: %{x}<br>Avg Duration: %{customdata[0]:.1f} sec<br>Meaningful Share: %{customdata[1]:.1f}%<extra></extra>"
    ), row=2, col=2)

    fig.add_annotation(
        text="Manager Performance",
        xref="x domain", yref="y domain", x=0.5, y=1.12,
        showarrow=False, font=dict(size=14, color=TH["text_color"], family="Arial"),
        row=2, col=2
    )

    fig.update_layout(
        showlegend=False,
        height=h, width=w,
        plot_bgcolor=TH["plot_bg"], paper_bgcolor=TH["paper_bg"],
        font=dict(color=TH["text_color"]),
        title=dict(text="Calls Overview Dashboard", x=0.5, xanchor="center",
                   font=dict(size=20, color=TH["text_color"]))
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)

    return fig, kpis

# ------------------------------------------------------------
# MARKETING DASHBOARD (OPTIMIZED + ENGLISH)
# ------------------------------------------------------------

def layout_marketing():
    main_data = DataManager.get_main()
    
    sources = ["All"] + sorted(main_data["Source"].dropna().unique().tolist())
    campaigns = ["All"] + sorted(main_data["Campaign"].dropna().unique().tolist())
    education_types = ["All"] + sorted(main_data["Education Type"].dropna().unique().tolist())
    months = sorted(main_data["Created Month"].dropna().unique().tolist())

    sidebar = html.Div(
        [
            html.H4("Filters", style={"marginTop": 0, "marginBottom": "10px"}),

            html.Label("Source:", style={"fontWeight": "600", "marginTop": "6px"}),
            dcc.Dropdown(sources, "All", id="marketing-source", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Campaign:", style={"fontWeight": "600"}),
            dcc.Dropdown(campaigns, "All", id="marketing-campaign", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Education Type:", style={"fontWeight": "600"}),
            dcc.Dropdown(education_types, "All", id="marketing-education", clearable=False, style={"marginBottom": "10px"}),

            html.Label("Period:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Div([
                html.Div([
                    html.Label("From:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[0], id="marketing-period-start", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block", "marginRight": "10px"}),

                html.Div([
                    html.Label("To:", style={"fontSize": "13px", "marginRight": "5px"}),
                    dcc.Dropdown(months, months[-1], id="marketing-period-end", clearable=False, style={"width": "110px"}),
                ], style={"display": "inline-block"})
            ], style={"marginBottom": "20px"}),
            html.Hr(),
            html.H4("Settings", style={"marginTop": "10px"}),

            html.Label("Sort:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[
                    {"label": "Ascending", "value": True},
                    {"label": "Descending", "value": False},
                ],
                value=False,
                id="marketing-sort",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Width:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[1000, 1200, 1300, 1500, 2000, 2500],
                value=1300,
                id="marketing-width",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            html.Label("Chart Height:", style={"fontWeight": "600"}),
            dcc.Dropdown(
                options=[600, 700, 800, 900, 1000],
                value=800,
                id="marketing-height",
                clearable=False,
                style={"marginBottom": "15px"},
            ),

            html.Hr(),
            html.H4("Dashboard Navigation", style={"marginTop": "10px", "marginBottom": "8px"}),
            get_switch_buttons("marketing"),
        ],
        style={
            "width": "320px",
            "padding": "16px",
            "backgroundColor": TH["theme_color"],
            "borderLeft": f"1px solid {TH['border']}",
            "position": "fixed",
            "right": 0,
            "top": 0,
            "height": "100vh",
            "overflowY": "auto",
            "color": TH["text_color"],
        },
    )

    content = html.Div(
        [
            html.Div(id="marketing-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
            dcc.Loading(
                id="loading-marketing",
                type="circle",
                children=dcc.Graph(id="marketing-figure", config={"displaylogo": False})
            )
        ],
        style={"marginRight": "340px", "padding": "16px", "backgroundColor": TH["paper_bg"], "color": TH["text_color"]}
    )

    return html.Div([content, sidebar], id="marketing-layout")

def compute_marketing_view(source, campaign, education_type, period_start, period_end, sort_asc, w, h):
    df_f = DataManager.get_main().copy()

    # Filters
    df_f = df_f[(df_f["Created Month"] >= period_start) & (df_f["Created Month"] <= period_end)]
    if source != "All": df_f = df_f[df_f["Source"] == source]
    if campaign != "All": df_f = df_f[df_f["Campaign"] == campaign]
    if education_type != "All": df_f = df_f[df_f["Education Type"] == education_type]

    # Aggregation and KPI calculations
    spend_level, unit, paid, prod = build_layers_no_inflation(df_f)
    AC_effective = get_effective_spend(spend_level, paid)
    k = compute_unit_kpis(spend_level, paid, df_f, cogs_dynamic=True, AC_override=AC_effective)
    product_metrics = calculate_product_unit_economics(df_f, AC_effective)

    # KPI cards
    kpis = [
        kpi_div("UA", f"{k['UA']:,}"),
        kpi_div("B", f"{k['B']:,}"),
        kpi_div("C1 (%)", f"{k['C1']:.1f}%"),
        kpi_div("AOV (€)", f"{k['AOV']:,.0f}"),
        kpi_div("CPA (€)", f"{k['CPA']:,.0f}"),
        kpi_div("CLTV (€)", f"{k['CLTV']:,.0f}"),
        kpi_div("ROI (%)", f"{k['ROI']:,.1f}%"),
        kpi_div("CM (€)", f"{k['CM']:,.0f}")
    ]

    # Charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Unit Economics (per customer)",
            "Marketing Spend Distribution",
            "Product Matrix (LTV vs CPA)",
            "Budget vs Revenue Share"
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.25
    )

    # 1) Unit Economics Breakdown
    rev_u  = _safe_div(k["Revenue"], k["B"]) if k["B"] > 0 else 0
    cogs_u = _safe_div(k["Total_COGS"], k["B"]) if k["B"] > 0 else 0
    ac_u   = _safe_div(k["AC"], k["B"]) if k["B"] > 0 else 0
    cm_u   = rev_u - cogs_u - ac_u

    x_labels = ["Revenue", "COGS", "Acquisition", "CM"]
    y_values = [rev_u, -cogs_u, -ac_u, cm_u]
    colors   = ["#4e79a7", "#e15759", "#f28e2b", "#59a14f"]

    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=colors,
        text=[f"{v:,.0f} €" for v in y_values],
        textposition="outside",
        cliponaxis=False,
        name="Unit Economics"
    ), row=1, col=1)

    fig.update_yaxes(title="€ per customer", row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#e15759", width=2, dash="dash"), row=1, col=1)

    # 2) Marketing Spend Distribution
    if k['B'] > 0 and product_metrics is not None:
        spend_by_product = (
            product_metrics[['Product', 'Marketing_Spend']]
            .sort_values('Marketing_Spend', ascending=False)
            .head(8)
        )
    
        fig.add_trace(go.Bar(
            x=spend_by_product['Product'],
            y=spend_by_product['Marketing_Spend'],
            marker_color=TH["accent_color"],
            name="Product Spend (€)",
            text=[f"{v:,.0f}€" for v in spend_by_product['Marketing_Spend']],
            textposition="outside",
            textfont=dict(size=10, color=TH["text_color"], family="Arial")
        ), row=1, col=2)
    
        fig.update_yaxes(title_text="Marketing Spend (€)", row=1, col=2, title_font=dict(size=12), tickfont=dict(size=12))
        fig.update_xaxes(title_text="Product", row=1, col=2, title_font=dict(size=12), tickfont=dict(size=10))

    # 3) Product Matrix (LTV vs CPA)
    if k['B'] > 0 and product_metrics is not None:
        fig.add_trace(go.Scatter(
            x=product_metrics['CPA'],
            y=product_metrics['LTV'],
            mode='markers+text',
            text=product_metrics['Product'],
            marker=dict(
                size=product_metrics['Revenue'] / 5000,
                color=product_metrics['ROI'],
                colorscale='RdYlGn',
                showscale=False,
                cmin=-50, cmax=100,
                line=dict(width=0),
                opacity=0.8
            ),
            textposition="middle center",
            textfont=dict(size=10, color=TH["text_color"], weight="bold"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CPA: %{x:,.0f} €<br>"
                "LTV: %{y:,.0f} €<br>"
                "ROI: %{marker.color:.0f}%<br>"
                "Revenue: %{marker.size:,.0f} €<extra></extra>"
            ),
        ), row=2, col=1)
    
        fig.update_xaxes(title_text="CPA (€)", row=2, col=1, title_font=dict(size=12), tickfont=dict(size=10))
        fig.update_yaxes(title_text="LTV (€)", row=2, col=1, title_font=dict(size=12), tickfont=dict(size=10))

    # 4) Budget vs Revenue Share
    if k['B'] > 0 and product_metrics is not None:
        product_metrics['Budget_Share'] = product_metrics['Marketing_Spend'] / AC_effective * 100
        product_metrics['Revenue_Share'] = product_metrics['Revenue'] / k['Revenue'] * 100
    
        min_size, max_size = 30, 90
        ltv_cac_values = product_metrics['LTV_CAC_Ratio']
        if ltv_cac_values.max() > ltv_cac_values.min():
            sizes = min_size + (ltv_cac_values - ltv_cac_values.min()) / \
                    (ltv_cac_values.max() - ltv_cac_values.min()) * (max_size - min_size)
        else:
            sizes = [80] * len(product_metrics)
    
        fig.add_trace(go.Scatter(
            x=product_metrics['Budget_Share'],
            y=product_metrics['Revenue_Share'],
            mode='markers+text',
            text=product_metrics['Product'],
            marker=dict(
                size=sizes,
                color=product_metrics['ROI'],
                colorscale='RdYlGn',
                showscale=False,
                cmin=-50, cmax=100,
                line=dict(width=0),
                opacity=0.85
            ),
            textposition="middle center",
            textfont=dict(size=12, color=TH["text_color"], weight="bold"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Budget Share: %{x:.1f}%<br>"
                "Revenue Share: %{y:.1f}%<br>"
                "LTV/CAC: %{customdata:.1f}x<br>"
                "ROI: %{marker.color:.0f}%<extra></extra>"
            ),
            customdata=product_metrics['LTV_CAC_Ratio'],
            showlegend=False
        ), row=2, col=2)
    
        fig.update_xaxes(title_text="Budget Share (%)", range=[0, 60], row=2, col=2, title_font=dict(size=12), tickfont=dict(size=10))
        fig.update_yaxes(title_text="Revenue Share (%)", range=[0, 60], row=2, col=2, title_font=dict(size=12), tickfont=dict(size=10))

    # Final styling
    fig.update_layout(
        height=h, width=w,
        plot_bgcolor=TH["plot_bg"], paper_bgcolor=TH["paper_bg"],
        font=dict(color=TH["text_color"]),
        title=dict(text="Marketing & ROI Dashboard", x=0.5, xanchor="center",
                   font=dict(size=20, color=TH["text_color"])),
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=TH["axis_line"], gridcolor=TH["grid_line"], zeroline=False)

    return fig, kpis

# ------------------------------------------------------------
# GEO DASHBOARD (OPTIMIZED + ENGLISH)
# ------------------------------------------------------------

def layout_geo():
    """Layout for GEO dashboard"""
    geo_data = prepare_geo_data_optimized()
    
    # Create filter lists from real data
    months = ["All"] + sorted(geo_data["Created Month"].unique().tolist())
    langs = ["All"] + sorted(geo_data["Deutsch_Level_Clean"].unique().tolist())
    sources = ["All"] + sorted(geo_data["Source_top"].dropna().unique().tolist())
    countries = ["All countries"] + sorted(geo_data['country'].unique().tolist())

    sidebar = html.Div([
        html.H4("Filters", style={"marginTop": 0, "marginBottom": "10px"}),

        html.Label("Country:", style={"fontWeight": "600"}),
        dcc.Dropdown(countries, "Germany", id="geo-country", clearable=False, style={"marginBottom": "10px"}),

        html.Label("Language Level:", style={"fontWeight": "600"}),
        dcc.Dropdown(langs, "All", id="geo-lang", clearable=False, style={"marginBottom": "10px"}),

        html.Label("Source:", style={"fontWeight": "600"}),
        dcc.Dropdown(sources, "All", id="geo-source", clearable=False, style={"marginBottom": "10px"}),

        html.Label("Month:", style={"fontWeight": "600"}),
        dcc.Dropdown(months, "All", id="geo-month", clearable=False, style={"marginBottom": "15px"}),

        html.Label("Map Mode:", style={"fontWeight": "600"}),
        dcc.Dropdown(
            options=[
                {"label": "🗺 Static", "value": "static"},
                {"label": "▶ Animation", "value": "anim"},
            ],
            value="static", id="geo-mode", clearable=False, style={"marginBottom": "15px"}
        ),

        html.Label("Map Style:", style={"fontWeight": "600"}),
        dcc.Dropdown(
            options=[
                {"label": "Light", "value": "carto-positron"},
                {"label": "Dark", "value": "carto-darkmatter"},
                {"label": "Street", "value": "open-street-map"},
            ],
            value="carto-positron", id="geo-map-style", clearable=False, style={"marginBottom": "15px"}
        ),

        html.Label("Map Width:", style={"fontWeight": "600"}),
        dcc.Dropdown([1000, 1200, 1300, 1500, 2000, 2500], 1500, id="geo-width", clearable=False, style={"marginBottom": "10px"}),

        html.Label("Map Height:", style={"fontWeight": "600"}),
        dcc.Dropdown([600, 700, 800, 900, 1000], 800, id="geo-height", clearable=False, style={"marginBottom": "15px"}),

        html.Hr(),
        html.H4("Dashboard Navigation", style={"marginTop": "10px", "marginBottom": "8px"}),
        get_switch_buttons("geo"),
    ], style={
        "width": "320px", "padding": "16px", "backgroundColor": TH["theme_color"],
        "borderLeft": f"1px solid {TH['border']}", "position": "fixed", "right": 0, "top": 0,
        "height": "100vh", "overflowY": "auto", "color": TH["text_color"]
    })

    content = html.Div([        
        html.Div(id="geo-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
        dcc.Loading(
            id="loading-geo",
            type="circle",
            children=dcc.Graph(id="geo-figure", config={"displaylogo": False})
        )
    ], style={"marginRight": "340px", "padding": "16px", "backgroundColor": TH["paper_bg"], "color": TH["text_color"]})

    return html.Div([content, sidebar], id="geo-layout")

def compute_geo_view_optimized(lang, source, month, mode, map_style, country, w, h):
    """Optimized function for GEO view computation"""
    
    # Use cached data
    df = prepare_geo_data_optimized()
    
    # COUNTRY FILTER
    if country != "All countries":
        df_clean = df[df['country'] == country]
    else:
        df_clean = df.copy()
    
    # Apply user filters
    if lang != "All": 
        df_clean = df_clean[df_clean["Deutsch_Level_Clean"] == lang]
    if source != "All": 
        df_clean = df_clean[df_clean["Source_top"] == source]
    if month != "All" and mode == "static": 
        df_clean = df_clean[df_clean["Created Month"] == month]

    # LIMIT POINTS COUNT (if needed)
    MAX_POINTS = 500
    if len(df_clean) > MAX_POINTS:
        # Take top cities by deal count
        top_cities = df_clean.groupby('City', observed=False)['deal_count'].sum().nlargest(MAX_POINTS).index
        df_clean = df_clean[df_clean['City'].isin(top_cities)]

    if df_clean.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data to display",
            paper_bgcolor=TH["paper_bg"],
            plot_bgcolor=TH["plot_bg"],
            font=dict(color=TH["text_color"]),
            height=h,
            width=w
        )
        return empty_fig, [kpi_div("No data", "—")]

    # FAST TOOLTIP - use pre-calculated data
    df_clean['custom_hover'] = (
        "<b>🏙 " + df_clean['City'].astype(str) + "</b><br>" +
        "🌍 Country: " + df_clean['country'].astype(str) + "<br>" +
        "📅 Period: " + ('All months' if month == 'All' else df_clean['Created Month'].astype(str)) + "<br>" +
        "📊 Deals: <b>" + df_clean['deal_count'].astype(str) + "</b><br>" +
        "✅ Successful: <b>" + df_clean['success_count'].astype(str) + "</b><br>" +
        "📈 Conversion: <b>" + df_clean['success_rate'].round(1).astype(str) + "%</b><br>" +
        "💰 Revenue: <b>" + df_clean['revenue'].round(0).astype(str) + " €</b><br>" +
        "🎓 Language: " + df_clean['Deutsch_Level_Clean'].astype(str) + "<br>" +
        "🔗 Source: " + df_clean['Source_top'].fillna('Not specified').astype(str)
    )

    # KPI blocks
    total_cities = df_clean["City"].nunique()
    total_deals = df_clean['deal_count'].sum()
    total_success = df_clean['success_count'].sum()
    total_revenue = df_clean['revenue'].sum()
    avg_success_rate = (total_success / total_deals * 100) if total_deals > 0 else 0

    kpis = [
        kpi_div("Cities", f"{total_cities:,}"),
        kpi_div("Deals", f"{total_deals:,}"),
        kpi_div("Successful", f"{total_success:,}"),
        kpi_div("Conversion", f"{avg_success_rate:.1f}%"),
        kpi_div("Revenue", f"{total_revenue:,.0f} €"),
    ]

    # MAP DATA PREPARATION
    df_clean = df_clean.copy()
    
    # Normalize circle sizes (based on deal count)
    max_deals = df_clean['deal_count'].max()
    if max_deals > 0:
        df_clean['scaled_size'] = 8 + (df_clean['deal_count'] / max_deals) * 25
    else:
        df_clean['scaled_size'] = 10

    animated = (mode == "anim")
    country_label = country if country != "All countries" else "all countries"
    period_label = "all months" if month == "All" else f"{month}"
    map_title = f"Deal Geography ({country_label}, {period_label})"

    # CREATE MAP
    if animated:
        fig = px.scatter_map(
            df_clean, 
            lat="lat", 
            lon="lng", 
            color="Deutsch_Level_Clean",
            size="scaled_size",
            hover_name="City",
            custom_data=['custom_hover'],
            animation_frame="Created Month", 
            zoom=3 if country == "All countries" else 5,
            title=map_title,
            map_style=map_style
        )
    else:
        fig = px.scatter_map(
            df_clean, 
            lat="lat", 
            lon="lng", 
            color="Deutsch_Level_Clean",
            size="scaled_size", 
            hover_name="City",
            custom_data=['custom_hover'],
            zoom=3 if country == "All countries" else 5,
            title=map_title,
            map_style=map_style
        )

    # TOOLTIP AND STYLING CONFIGURATION
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(
            sizemin=6,
            opacity=0.7,
            allowoverlap=False
        )
    )

    # Map centering
    if not df_clean.empty:
        if country == "All countries":
            fig.update_layout(map=dict(center=dict(lat=30, lon=20), zoom=2))
        else:
            avg_lat = df_clean['lat'].mean()
            avg_lon = df_clean['lng'].mean()
            fig.update_layout(map=dict(center=dict(lat=avg_lat, lon=avg_lon), zoom=5))

    # FINAL SETTINGS
    fig.update_layout(
        height=h, width=w,
        margin=dict(l=10, r=10, t=80, b=10),
        hoverlabel=dict(
            bgcolor="white", 
            font_size=12, 
            font_family="Arial", 
            font_color="black", 
            align="left"
        ),
        legend=dict(
            title="🎓 Language Level",
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        paper_bgcolor=TH["paper_bg"], 
        plot_bgcolor=TH["plot_bg"],
        font=dict(color=TH["text_color"]),
        title={
            'text': map_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': TH["text_color"]}
        }
    )
    
    return fig, kpis

# ------------------------------------------------------------
# UNIFIED SWITCH BUTTONS
# ------------------------------------------------------------

def get_switch_buttons(current_dashboard):
    """Returns dashboard navigation buttons"""
    if current_dashboard == "deals":
        return html.Div([
            html.Button("Deals Dashboard", id="to-deals", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#4e79a7", "color": "white", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Calls Dashboard", id="to-calls", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Marketing Dashboard", id="to-marketing", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Geo Dashboard", id="to-geo", n_clicks=0, 
                        style={"width": "100%", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"})
        ], style={"marginBottom": "12px"})
    elif current_dashboard == "calls":
        return html.Div([
            html.Button("Deals Dashboard", id="to-deals", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Calls Dashboard", id="to-calls", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#4e79a7", "color": "white", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Marketing Dashboard", id="to-marketing", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Geo Dashboard", id="to-geo", n_clicks=0,  
                        style={"width": "100%", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"})
        ], style={"marginBottom": "12px"})
    elif current_dashboard == "marketing":
        return html.Div([
            html.Button("Deals Dashboard", id="to-deals", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Calls Dashboard", id="to-calls", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Marketing Dashboard", id="to-marketing", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#4e79a7", "color": "white", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Geo Dashboard", id="to-geo", n_clicks=0,  
                        style={"width": "100%", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"})
        ], style={"marginBottom": "12px"})
    else:  # geo
        return html.Div([
            html.Button("Deals Dashboard", id="to-deals", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Calls Dashboard", id="to-calls", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Marketing Dashboard", id="to-marketing", n_clicks=0,
                        style={"width": "100%", "marginBottom": "8px", "backgroundColor": "#e5e7eb", "color": "#2c3e50", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"}),
            html.Button("Geo Dashboard", id="to-geo", n_clicks=0,
                        style={"width": "100%", "backgroundColor": "#4e79a7", "color": "white", "border": "none", "padding": "8px", "borderRadius": "5px", "cursor": "pointer"})
        ], style={"marginBottom": "12px"})

# ------------------------------------------------------------
# DASH APP INITIALIZATION
# ------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Unified Dashboard Suite"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id="ROOT")
])

@app.callback(
    Output("ROOT", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/calls":
        return layout_calls()
    elif pathname == "/marketing":
        return layout_marketing()
    elif pathname == "/geo":  
        return layout_geo()
    else:
        return layout_deals()

@app.callback(
    Output("url", "pathname"),
    Input("to-calls", "n_clicks"),
    Input("to-deals", "n_clicks"),
    Input("to-marketing", "n_clicks"),
    Input("to-geo", "n_clicks"),
    prevent_initial_call=True
)
def update_url(n_calls, n_deals, n_marketing, n_geo):
    trig = ctx.triggered_id
    if trig == "to-calls": return "/calls"
    elif trig == "to-deals": return "/deals"
    elif trig == "to-marketing": return "/marketing"
    elif trig == "to-geo": return "/geo"
    return "/deals"

# Callbacks for all dashboards
@app.callback(
    Output("deal-figure", "figure"),
    Output("deal-kpis", "children"),
    Input("deal-manager", "value"),
    Input("deal-stage", "value"),
    Input("deal-source", "value"),
    Input("deal-campaign", "value"),
    Input("deal-period-start", "value"),
    Input("deal-period-end", "value"),
    Input("deal-metric", "value"),
    Input("deal-sort", "value"),
    Input("deal-width", "value"),
    Input("deal-height", "value"),
    Input("deal-funnel-type", "value"),
)
def update_deals(manager, stage, source, campaign,
                 period_start, period_end,
                 metric, sort_asc, w, h, funnel_kind):
    fig, kpis = compute_deals_view(manager, stage, source, campaign,
                                    period_start, period_end,
                                    metric, sort_asc, w, h, funnel_kind)
    return fig, kpis

@app.callback(
    Output("calls-figure", "figure"),
    Output("calls-kpis", "children"),
    Input("calls-manager", "value"),
    Input("calls-type", "value"),
    Input("calls-meaning", "value"),
    Input("calls-period-start", "value"),
    Input("calls-period-end", "value"),
    Input("calls-time-mode", "value"),
    Input("calls-sort", "value"),
    Input("calls-width", "value"),
    Input("calls-height", "value"),
)
def update_calls(manager, ctype, meaning, period_start, period_end, time_mode, sort_asc, w, h):
    fig, kpis = compute_calls_view(manager, ctype, meaning, period_start, period_end, time_mode, sort_asc, w, h)
    return fig, kpis

@app.callback(
    Output("marketing-figure", "figure"),
    Output("marketing-kpis", "children"),
    Input("marketing-source", "value"),
    Input("marketing-campaign", "value"),
    Input("marketing-education", "value"),
    Input("marketing-period-start", "value"),
    Input("marketing-period-end", "value"),
    Input("marketing-sort", "value"),
    Input("marketing-width", "value"),
    Input("marketing-height", "value"),
)
def update_marketing(source, campaign, education_type, period_start, period_end, sort_asc, w, h):
    fig, kpis = compute_marketing_view(source, campaign, education_type, period_start, period_end, sort_asc, w, h)
    return fig, kpis

@app.callback(
    Output("geo-figure", "figure"),
    Output("geo-kpis", "children"),
    Input("geo-lang", "value"),
    Input("geo-source", "value"),
    Input("geo-month", "value"),
    Input("geo-mode", "value"),
    Input("geo-map-style", "value"),
    Input("geo-country", "value"),
    Input("geo-width", "value"),
    Input("geo-height", "value"),
)
def update_geo(lang, source, month, mode, map_style, country, w, h):
    try:
        fig, kpis = compute_geo_view_optimized(lang, source, month, mode, map_style, country, w, h)
        return fig, kpis
    except Exception as e:
        print(f"Error in GEO dashboard: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        
        # Return empty figure in case of error
        error_fig = go.Figure()
        error_fig.update_layout(
            title="Error building map",
            paper_bgcolor=TH["paper_bg"],
            plot_bgcolor=TH["plot_bg"],
            font=dict(color=TH["text_color"]),
            height=h,
            width=w
        )
        error_kpis = [kpi_div("Error", "Check console")]
        
        return error_fig, error_kpis


# ============================================================
# RUN APP
# ============================================================
server = app.server

if __name__ == "__main__":
    app.run( debug=False, host="0.0.0.0", port=8050)