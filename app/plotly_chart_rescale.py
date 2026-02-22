"""
Render a Plotly figure in an HTML iframe with a plotly_relayout handler that
rescales the price y-axis to the visible x range when the user zooms or pans.
Used for the Technical Chart so drag-zoom keeps the chart readable.
"""


def render_plotly_chart_with_y_rescale(fig, config: dict | None = None) -> str:
    """
    Build an HTML string that renders the Plotly figure and attaches a
    plotly_relayout listener to rescale the first subplot's y-axis (price panel)
    to the visible x range when the user zooms or pans.

    fig: plotly.graph_objects.Figure (e.g. from create_bloomberg_chart)
    config: optional Plotly config dict (scrollZoom, displayModeBar, etc.)

    Returns HTML string for st.components.v1.html(..., height=850).
    """
    import json
    fig_json = fig.to_json()
    # Prevent </script> in JSON from closing the script tag in HTML
    fig_json_escaped = fig_json.replace("</", "<\\/")
    config = config or {}
    config_json = json.dumps(config)

    html = f'''<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <meta charset="utf-8">
  <style>body {{ margin: 0; background: #0d1117; }}</style>
</head>
<body style="background:#0d1117;">
  <div id="chart" style="width:100%;height:850px;"></div>
  <script id="fig-data" type="application/json">{fig_json_escaped}</script>
  <script id="config-data" type="application/json">{config_json}</script>
  <script>
(function() {{
  var figEl = document.getElementById('fig-data');
  var configEl = document.getElementById('config-data');
  var fig = JSON.parse(figEl.textContent);
  var config = JSON.parse(configEl.textContent);

  var gd = document.getElementById('chart');
  Plotly.newPlot(gd, fig.data, fig.layout, config);

  function toMs(v) {{
    if (v == null) return NaN;
    if (typeof v === 'number') return v;
    var d = new Date(v);
    return isNaN(d.getTime()) ? NaN : d.getTime();
  }}

  function getYValuesInRange(trace, x0, x1) {{
    var xs = trace.x || [];
    var len = xs.length;
    var out = [];
    var i;
    if (trace.type === 'candlestick') {{
      var high = trace.high || [];
      var low = trace.low || [];
      for (i = 0; i < len; i++) {{
        var xi = toMs(xs[i]);
        if (!isNaN(xi) && xi >= x0 && xi <= x1) {{
          if (high[i] != null) out.push(Number(high[i]));
          if (low[i] != null) out.push(Number(low[i]));
        }}
      }}
    }} else {{
      var ys = trace.y || [];
      for (i = 0; i < len; i++) {{
        var xi = toMs(xs[i]);
        if (!isNaN(xi) && xi >= x0 && xi <= x1 && ys[i] != null)
          out.push(Number(ys[i]));
      }}
    }}
    return out;
  }}

  gd.on('plotly_relayout', function(eventData) {{
    var xRange = eventData['xaxis.range'];
    if (!xRange || !Array.isArray(xRange) || xRange.length < 2) return;
    var x0 = toMs(xRange[0]);
    var x1 = toMs(xRange[1]);
    if (isNaN(x0) || isNaN(x1)) return;

    var allY = [];
    for (var i = 0; i < fig.data.length; i++) {{
      var trace = fig.data[i];
      var yaxis = trace.yaxis || 'y';
      if (yaxis !== 'y') continue;
      var ys = getYValuesInRange(trace, x0, x1);
      allY = allY.concat(ys);
    }}
    if (allY.length === 0) return;
    var yMin = Math.min.apply(null, allY);
    var yMax = Math.max.apply(null, allY);
    var pad = (yMax - yMin) * 0.02 || 1;
    yMin = yMin - pad;
    yMax = yMax + pad;
    Plotly.relayout(gd, {{ 'yaxis.range': [yMin, yMax] }});
  }});
}})();
  </script>
</body>
</html>'''
    return html
