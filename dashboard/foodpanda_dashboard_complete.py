import pandas as pd
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc

# --- STYLING & SETUP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap'])
app.title = "Foodpanda Market Basket Dashboard"

colors = {
    'background': '#f8f9fa', 'text': '#495057', 'primary': '#E32D3A', 'card_bg': '#FFFFFF',
    'accent': '#007bff', 'success': '#28a745', 'danger': '#dc3545'
}

# --- DATA LOADING & CLEANING ---
try:
    file_path = '/home/prathmk85/Foodpanda_MarketBasket_Project/data/Foodpanda Analysis Dataset.csv'
    df_raw = pd.read_csv(file_path)
    city_mapping = {'Peshawar': 'Pune', 'Multan': 'Mumbai', 'Lahore': 'Lucknow', 'Karachi': 'Kochi', 'Islamabad': 'Indore'}
    df_raw['city'] = df_raw['city'].replace(city_mapping)
    df_raw['order_date'] = pd.to_datetime(df_raw['order_date'], format='%m/%d/%Y', errors='coerce')
    df_raw['signup_date'] = pd.to_datetime(df_raw['signup_date'], format='%m/%d/%Y', errors='coerce')
    df_raw = df_raw.dropna(subset=['order_date'])
    df_raw['revenue'] = (pd.to_numeric(df_raw['price'], errors='coerce').fillna(0) * pd.to_numeric(df_raw['quantity'], errors='coerce').fillna(1))
    print(f"Loaded {len(df_raw)} records successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    df_raw = pd.DataFrame()

if not df_raw.empty and df_raw['order_date'].notna().any():
    min_date, max_date = df_raw['order_date'].min().date(), df_raw['order_date'].max().date()
else:
    min_date, max_date = datetime.now().date(), datetime.now().date()

# --- FOOTER DEFINITION ---
# List of contributors with their GitHub usernames
contributors = [
    {"name": "Prathmesh Kulkarni", "username": "prathmk85"},
    {"name": "Azhan Khan", "username": "AzhanK101"},
]

contributor_cards = []
for contributor in contributors:
    github_url = f"https://github.com/{contributor['username']}"
    avatar_url = f"https://github.com/{contributor['username']}.png?size=80"
    card = dbc.Col(
        html.A(
            href=github_url,
            target="_blank",
            children=[
                html.Div([
                    html.Img(src=avatar_url, style={'borderRadius': '50%', 'width': '80px'}),
                    html.P(contributor['name'], className="mt-2 mb-0", style={'fontWeight': 'bold'}),
                ], className="text-center p-2"),
            ],
            style={'textDecoration': 'none', 'color': 'inherit'}
        ),
        width="auto"
    )
    contributor_cards.append(card)

footer = html.Footer([
    html.Hr(className="my-4"),
    dcc.Markdown("Made with ❤️ using Dash", className="text-center"),
    html.H5("Project Contributors", className="text-center mt-4 mb-3"),
    dbc.Row(contributor_cards, justify="center", className="g-3")
], className="mt-5")


# --- LAYOUT DEFINITION ---
app.layout = dbc.Container(fluid=True, style={'backgroundColor': colors['background'], 'fontFamily': 'Lato, sans-serif'}, children=[
    dbc.Row(dbc.Col(html.H1("Foodpanda Market Basket Analysis", className="text-center my-4", style={'color': colors['primary'], 'fontWeight': 'bold'}))),
    dbc.Row([
        dbc.Col(dcc.DatePickerRange(id='filter-date', min_date_allowed=min_date, max_date_allowed=max_date, start_date=min_date, end_date=max_date), width=12, lg=4, className="mb-3"),
        dbc.Col(dcc.Dropdown(id='filter-city', options=[{'label': c, 'value': c} for c in sorted(df_raw['city'].dropna().unique())] if not df_raw.empty else [], multi=True, value=list(df_raw['city'].dropna().unique()) if not df_raw.empty else [], placeholder="Select Cities"), width=12, lg=8, className="mb-3"),
    ], className="p-3 bg-white rounded shadow-sm mb-4"),
    dbc.Row(id='kpi-cards', className="mb-4", justify="start"),
    dbc.Tabs(id='main-tabs', children=[
        dbc.Tab(label='Market Basket Analysis', tab_id='tab-basket', children=[dbc.Row([dbc.Col(dcc.Graph(id='assoc-network'), width=12, className="mt-4"), dbc.Col(dash_table.DataTable(id='tbl-rules', columns=[{"name": "Rule", "id": "rule"}, {"name": "Support", "id": "support", "type": "numeric", "format": {'specifier': ".3f"}}, {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {'specifier': ".3f"}}, {"name": "Lift", "id": "lift", "type": "numeric", "format": {'specifier': ".3f"}}], page_size=10, style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'fontFamily': 'Lato'}, style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}), width=12, className="my-4")])]),
        dbc.Tab(label='Customer Insights', tab_id='tab-customer', children=[dbc.Row([dbc.Col(dcc.Graph(id='graph-customer-gender'), width=12, lg=4), dbc.Col(dcc.Graph(id='graph-customer-age'), width=12, lg=4), dbc.Col(dcc.Graph(id='graph-customer-city'), width=12, lg=4)], className="mt-4")]),
        dbc.Tab(label='Order Trends', tab_id='tab-ordertrend', children=[dbc.Row([dbc.Col(dcc.Graph(id='graph-order-trend'), width=12)], className="mt-4"), dbc.Row([dbc.Col(dcc.Graph(id='graph-restaurant-popularity'), width=12, lg=6), dbc.Col(dcc.Graph(id='graph-popular-dishes'), width=12, lg=6)], className="mt-4")])
    ]),
    
    # Add the footer to the bottom of the layout
    footer
])

# --- HELPER FUNCTIONS & CALLBACKS ---
def create_powerbi_kpi(title, value_str, change_str, change_color, chart_fig):
    return dbc.Col(dbc.Card(dbc.CardBody([html.P(title, className="card-subtitle text-muted"), html.H3(value_str, className="card-title my-2", style={'fontWeight': 'bold'}), html.P(change_str, className="card-text", style={'color': change_color, 'minHeight': '24px'}), dcc.Graph(figure=chart_fig, config={'displayModeBar': False}, style={'height': '60px'})]), className="shadow-sm border-0 h-100"), width=12, sm=6, md=4, lg=4, className="mb-4")

def apply_chart_style(fig, title):
    fig.update_layout(title=title, plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color=colors['text'], title_font_size=20, margin=dict(l=40, r=20, t=60, b=40))
    return fig

def format_change_indicator(current_val, prev_val):
    if prev_val > 0:
        change = (current_val - prev_val) / prev_val
        icon = "▲" if change >= 0 else "▼"
        color = colors['success'] if change >= 0 else colors['danger']
        return f"{icon} {change:.1%}", color
    elif current_val > 0: return "▲ (New)", colors['success']
    else: return "—", colors['text']

@app.callback(Output('kpi-cards', 'children'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_kpi_cards(start_date, end_date, selected_cities):
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    duration = end_dt - start_dt
    prev_end_dt = start_dt - timedelta(days=1)
    prev_start_dt = prev_end_dt - duration
    df_current = safe_filter_data(start_date, end_date, selected_cities)
    df_previous = safe_filter_data(prev_start_dt.date(), prev_end_dt.date(), selected_cities)
    if df_current.empty: return dbc.Col(html.P("No data for the selected filters."), className="text-center")
    orders_current, orders_previous, orders_goal = len(df_current), len(df_previous), 10000
    rev_current, rev_previous = df_current['revenue'].sum(), df_previous['revenue'].sum()
    aov_current = rev_current / orders_current if orders_current else 0
    aov_previous = rev_previous / orders_previous if orders_previous else 0
    unique_customers = df_current['customer_id'].nunique()
    avg_rating = df_current['rating'].mean() if df_current['rating'].notna().any() else 0
    donut_fig = go.Figure(go.Pie(values=[orders_current, max(0, orders_goal - orders_current)], hole=0.7, marker_colors=[colors['primary'], '#eeeeee'], textinfo='none', hoverinfo='none', direction='clockwise', sort=False))
    donut_fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
    rev_trend = df_current.groupby(df_current['order_date'].dt.date)['revenue'].sum()
    spark_rev_fig = go.Figure(go.Scatter(x=rev_trend.index, y=rev_trend.values, fill='tozeroy', mode='lines', line=dict(color=colors['success'], width=2)))
    spark_rev_fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
    aov_trend = df_current.groupby(df_current['order_date'].dt.date)['revenue'].sum() / df_current.groupby(df_current['order_date'].dt.date).size()
    spark_aov_fig = go.Figure(go.Scatter(x=aov_trend.index, y=aov_trend.values, fill='tozeroy', mode='lines', line=dict(color=colors['accent'], width=2)))
    spark_aov_fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
    kpis = []
    orders_change_str, orders_color = format_change_indicator(orders_current, orders_previous)
    kpis.append(create_powerbi_kpi("Total Orders", f"{orders_current:,}", orders_change_str, orders_color, donut_fig))
    rev_change_str, rev_color = format_change_indicator(rev_current, rev_previous)
    kpis.append(create_powerbi_kpi("Total Revenue", f"₹{rev_current:,.0f}", rev_change_str, rev_color, spark_rev_fig))
    aov_change_str, aov_color = format_change_indicator(aov_current, aov_previous)
    kpis.append(create_powerbi_kpi("Avg Order Value", f"₹{aov_current:.0f}", aov_change_str, aov_color, spark_aov_fig))
    kpis.append(dbc.Col(dbc.Card(dbc.CardBody([html.P("Unique Customers"), html.H2(f"{unique_customers:,}"), html.P("vs Previous Period", className="text-muted")]), className="h-100 shadow-sm border-0"), width=12, sm=6, md=4, lg=4, className="mb-4"))
    kpis.append(dbc.Col(dbc.Card(dbc.CardBody([html.P("Avg Rating"), html.H2(f"{avg_rating:.2f} ⭐"), html.P("All Time Average", className="text-muted")]), className="h-100 shadow-sm border-0"), width=12, sm=6, md=4, lg=4, className="mb-4"))
    return kpis

def create_empty_figure(title):
    fig = go.Figure(); fig.add_annotation(text="No data for selected filters.", x=0.5, y=0.5, showarrow=False)
    return apply_chart_style(fig, title)
def safe_filter_data(start_date, end_date, selected_cities):
    if df_raw.empty: return pd.DataFrame()
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if not selected_cities: selected_cities = list(df_raw['city'].dropna().unique())
    return df_raw[(df_raw['order_date'] >= start_dt) & (df_raw['order_date'] <= end_dt) & (df_raw['city'].isin(selected_cities))]

@app.callback([Output('assoc-network', 'figure'), Output('tbl-rules', 'data')], [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_market_basket(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure("Top 10 Most Frequent Items"), []
    item_counts = df['dish_name'].value_counts().head(10).reset_index(); item_counts.columns = ['Dish', 'Count']
    fig = px.bar(item_counts, x='Dish', y='Count', color_discrete_sequence=[colors['primary']])
    table_data = [{'rule': f"Item: {row['Dish']}", 'support': row['Count'] / len(df), 'confidence': 1.0, 'lift': 1.0} for _, row in item_counts.head(5).iterrows()]
    return apply_chart_style(fig, 'Top 10 Most Frequent Items'), table_data
@app.callback(Output('graph-customer-gender', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_customer_gender(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty or df['gender'].isna().all(): return create_empty_figure('Customer Distribution by Gender')
    counts = df['gender'].value_counts()
    fig = px.pie(values=counts.values, names=counts.index)
    return apply_chart_style(fig, 'Customer Distribution by Gender')
@app.callback(Output('graph-customer-age', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_customer_age(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure('Customer Distribution by Age Group')
    counts_df = df['age'].value_counts().reset_index(); counts_df.columns = ['Age Group', 'Number of Orders']
    fig = px.bar(counts_df, x='Age Group', y='Number of Orders', color_discrete_sequence=[colors['accent']])
    return apply_chart_style(fig, 'Customer Distribution by Age Group')
@app.callback(Output('graph-customer-city', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_customer_city(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure('Top 10 Cities by Orders')
    counts_df = df['city'].value_counts().head(10).reset_index(); counts_df.columns = ['City', 'Number of Orders']
    fig = px.bar(counts_df, x='Number of Orders', y='City', orientation='h', color_discrete_sequence=[colors['primary']])
    return apply_chart_style(fig, 'Top 10 Cities by Orders')
@app.callback(Output('graph-order-trend', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_order_trends(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure('Daily Order Trends')
    daily_orders = df.groupby(df['order_date'].dt.date).size().reset_index(name='orders')
    fig = px.line(daily_orders, x='order_date', y='orders', color_discrete_sequence=[colors['primary']])
    return apply_chart_style(fig, 'Daily Order Trends')
@app.callback(Output('graph-restaurant-popularity', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_restaurant_popularity(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure('Top 10 Restaurants by Orders')
    counts_df = df['restaurant_name'].value_counts().head(10).reset_index(); counts_df.columns = ['Restaurant', 'Number of Orders']
    fig = px.bar(counts_df, x='Restaurant', y='Number of Orders', color_discrete_sequence=[colors['accent']])
    return apply_chart_style(fig, 'Top 10 Restaurants by Orders')
@app.callback(Output('graph-popular-dishes', 'figure'), [Input('filter-date', 'start_date'), Input('filter-date', 'end_date'), Input('filter-city', 'value')])
def update_popular_dishes(start_date, end_date, selected_cities):
    df = safe_filter_data(start_date, end_date, selected_cities)
    if df.empty: return create_empty_figure('Top 10 Most Popular Dishes')
    popularity_df = df.groupby('dish_name')['quantity'].sum().nlargest(10).reset_index(); popularity_df.columns = ['Dish Name', 'Total Quantity Sold']
    fig = px.bar(popularity_df, x='Total Quantity Sold', y='Dish Name', orientation='h', color_discrete_sequence=[colors['primary']])
    return apply_chart_style(fig, 'Top 10 Most Popular Dishes')

if __name__ == '__main__':
    app.run(debug=True, port=8050)