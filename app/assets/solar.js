/*
 * Browser-side two-body propagation for the Near-Earth Objects view.
 *
 * Same model as core/kepler.py: Kepler's equation by Newton iteration, then
 * r = a (cos E - e) P + a sqrt(1 - e^2) sin E Q from each orbit's pack
 * (built by app/neo_figures.py:orbit_pack). Units: AU, days, radians.
 * Checked against core.kepler.position_at in tests/test_neo_page.py.
 */
(function () {
  const TWO_PI = 2 * Math.PI;

  function keplerPosition(p, k, jd) {
    const a = p.a[k], e = p.e[k];
    let M = (p.M0[k] + p.n[k] * (jd - p.ep[k])) % TWO_PI;
    if (M < 0) { M += TWO_PI; }
    let E = e > 0.8 ? Math.PI : M + e * Math.sin(M);
    for (let it = 0; it < 50; it++) {
      const d = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
      E -= d;
      if (Math.abs(d) < 1e-12) { break; }
    }
    const x = a * (Math.cos(E) - e), y = a * Math.sqrt(1 - e * e) * Math.sin(E);
    return [x * p.Px[k] + y * p.Qx[k], x * p.Py[k] + y * p.Qy[k], x * p.Pz[k] + y * p.Qz[k]];
  }

  function positions(p, jd) {
    const n = p.a.length, xs = new Array(n), ys = new Array(n), zs = new Array(n);
    for (let k = 0; k < n; k++) {
      const r = keplerPosition(p, k, jd);
      xs[k] = r[0]; ys[k] = r[1]; zs[k] = r[2];
    }
    return [xs, ys, zs];
  }

  function jdToDate(jd) {
    return new Date((jd - 2440587.5) * 86400000).toISOString().slice(0, 10);
  }

  window.orbitLabSolar = { keplerPosition: keplerPosition, positions: positions, jdToDate: jdToDate };

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    solar: {
      tick: function (n, day, max) {
        return day >= max ? 0 : Math.min(day + 4, max);
      },

      togglePlay: function (n, disabled) {
        return [!disabled, disabled ? "❚❚" : "▶"];
      },

      render: function (day, scene) {
        if (!scene) { return ""; }
        const jd = scene.t0 + (day || 0);
        const gd = document.querySelector("#solar-graph .js-plotly-plot");
        if (gd && gd.data) {
          const idx = function (uid) { return gd.data.findIndex(function (t) { return t.uid === uid; }); };
          const update = { x: [], y: [], z: [] }, traces = [];
          const add = function (uid, xyz) {
            const i = idx(uid);
            if (i < 0) { return; }
            update.x.push(xyz[0]); update.y.push(xyz[1]); update.z.push(xyz[2]); traces.push(i);
          };
          add("planets", positions(scene.planets, jd));
          if (scene.target) { add("target", positions(scene.target, jd)); }
          if (scene.cloud && scene.cloud.a.length) { add("cloud", positions(scene.cloud, jd)); }
          if (scene.craft) {
            const c = scene.craft, t = jd - c.dep_jd;
            if (t >= 0 && t <= c.tof) {
              const f = t / c.tof * (c.x.length - 1), i0 = Math.floor(f), i1 = Math.min(i0 + 1, c.x.length - 1), w = f - i0;
              const lerp = function (arr) { return arr[i0] * (1 - w) + arr[i1] * w; };
              add("craft", [[lerp(c.x)], [lerp(c.y)], [lerp(c.z)]]);
            } else {
              add("craft", [[null], [null], [null]]);
            }
          }
          if (traces.length) { Plotly.restyle(gd, update, traces); }
        }
        return jdToDate(jd);
      }
    }
  });
})();
