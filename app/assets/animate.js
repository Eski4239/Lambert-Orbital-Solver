/*
 * Browser-side animation for the Earth-orbit view.
 *
 * The server sends the orbit once (figure + sampled track). Scrubbing or
 * playing the time slider only moves two traces here, the satellite (uid
 * "sat") and, in the inertial 3D view, the coastlines (uid "coast"), which
 * rotate with Greenwich sidereal time. No Python round-trip per frame.
 */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  orbit: {
    tick: function (n, k, track) {
      if (!track) { return 0; }
      return (k + 1) % track.t.length;
    },

    togglePlay: function (n, disabled) {
      const playing = disabled;  // was stopped -> now playing
      return [!disabled, playing ? "❚❚" : "▶"];
    },

    render: function (k, track, view, frame, coast) {
      if (!track) { return ""; }
      k = Math.min(k || 0, track.t.length - 1);
      const gd = document.querySelector("#orbit-graph .js-plotly-plot");
      if (gd && gd.data) {
        const sat = gd.data.findIndex(function (t) { return t.uid === "sat"; });
        if (view === "ground") {
          if (sat >= 0) { Plotly.restyle(gd, { x: [[track.lon[k]]], y: [[track.lat[k]]] }, [sat]); }
        } else {
          const src = frame === "eci" ? track.eci : track.ecef;
          if (sat >= 0) {
            Plotly.restyle(gd, { x: [[src.x[k]]], y: [[src.y[k]]], z: [[src.z[k]]] }, [sat]);
          }
          const ci = gd.data.findIndex(function (t) { return t.uid === "coast"; });
          const ei = gd.data.findIndex(function (t) { return t.uid === "earth"; });
          if (frame === "eci" && ci >= 0 && ei >= 0 && coast) {
            // ECEF -> ECI: rotate the globe and coastlines about Z by +GMST.
            const th = track.gmst[k] * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
            const n = coast.x.length, xs = new Array(n), ys = new Array(n);
            for (let i = 0; i < n; i++) {
              const x = coast.x[i], y = coast.y[i];
              if (x === null) { xs[i] = null; ys[i] = null; continue; }
              xs[i] = c * x - s * y; ys[i] = s * x + c * y;
            }
            const m = coast.earth_x.length, ex = new Array(m), ey = new Array(m);
            for (let i = 0; i < m; i++) {
              const x = coast.earth_x[i], y = coast.earth_y[i];
              ex[i] = c * x - s * y; ey[i] = s * x + c * y;
            }
            Plotly.restyle(gd, { x: [xs, ex], y: [ys, ey] }, [ci, ei]);
          }
        }
      }
      const t = track.t[k];
      const hh = Math.floor(t / 3600), mm = Math.floor((t % 3600) / 60), ss = Math.floor(t % 60);
      const pad = function (v) { return String(v).padStart(2, "0"); };
      const alt = Math.round(track.alt[k]).toLocaleString("en-US");
      return "T+" + pad(hh) + ":" + pad(mm) + ":" + pad(ss) + "  ·  alt " + alt + " km";
    }
  }
});
