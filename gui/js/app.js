// iPad mode, step 1: the planet with our locations. Tap a location to fly to it and read about it.

import { Globe } from "./globe.js";

const $ = (sel) => document.querySelector(sel);
const side = $("#side");
const list = $("#list");
const hint = $("#hint");

const fmtLat = (v) => `${Math.abs(v).toFixed(2)}° ${v < 0 ? "S" : "N"}`;
const fmtLon = (v) => `${Math.abs(v).toFixed(2)}° ${v < 0 ? "W" : "E"}`;

const isLandscape = () => innerWidth > 760 && innerWidth >= innerHeight;

// The part of the screen the planet may use: right of the text, or above it in portrait.
function freeArea() {
  const r = side.getBoundingClientRect();
  if (isLandscape()) return { left: r.right + 28, top: 0, right: innerWidth, bottom: innerHeight };
  return { left: 0, top: 0, right: innerWidth, bottom: r.top };
}

function locationItem(loc) {
  const li = document.createElement("li");
  li.className = "loc";
  li.dataset.id = loc.id;
  li.innerHTML = `
    <button class="loc-head" aria-expanded="false">
      <span class="loc-name"></span>
      <span class="loc-place"></span>
    </button>
    <div class="loc-body"><div>
      <p></p>
      <p></p>
      <a class="go off" href="#">Looking for the map</a>
    </div></div>`;
  li.querySelector(".loc-name").textContent = loc.name;
  li.querySelector(".loc-place").textContent = `${loc.place}, ${fmtLat(loc.lat)} ${fmtLon(loc.lon)}`;
  const [summary, detail] = li.querySelectorAll("p");
  summary.textContent = loc.summary;
  detail.textContent = loc.detail;
  checkViewer(loc, li.querySelector(".go"));
  return li;
}

// The link opens the 3D map of the location when it has been built (scripts/view3d.py).
async function checkViewer(loc, a) {
  let ok = false;
  try {
    ok = (await fetch(loc.viewer, { method: "HEAD" })).ok;
  } catch { /* not served */ }
  a.textContent = ok ? "Open the map →" : "The map of this place is not built yet";
  a.classList.toggle("off", !ok);
  if (ok) a.href = loc.viewer;
}

async function main() {
  const { locations, features = [] } = await (await fetch("locations.json")).json();
  locations.forEach((loc) => list.append(locationItem(loc)));

  let selected = null;
  const globe = new Globe($("#globe"), $("#labels"), {
    texture: "assets/mars_4k.jpg",
    freeArea,
    onReady: () => document.body.classList.add("ready"),
    onSelect: (id) => select(id === selected ? null : id),
  });
  features.forEach((f) => globe.addFeature(f));
  locations.forEach((loc) => globe.addPin(loc));

  function select(id) {
    selected = id;
    for (const li of list.children) {
      const on = li.dataset.id === id;
      li.classList.toggle("is-open", on);
      li.querySelector(".loc-head").setAttribute("aria-expanded", on);
    }
    if (id) globe.focus(id);
    else globe.release();
  }

  list.addEventListener("click", (e) => {
    const head = e.target.closest(".loc-head");
    if (head) select(head.parentElement.dataset.id === selected ? null : head.parentElement.dataset.id);
  });
  addEventListener("keydown", (e) => e.key === "Escape" && select(null));
  addEventListener("globe-touch", () => hint.classList.add("gone"), { once: true });

  // The text block changes size when a place opens (in portrait that moves the planet).
  new ResizeObserver(() => globe.resize()).observe(side);
  window.__gui = { globe, select };
}

main();
