// iPad mode, step 1: the planet with our locations. Pick a location in the list or on the
// globe to fly to it and read about it; "Open 3D map" opens the map of the site.

import { Globe } from "./globe.js";

const $ = (sel) => document.querySelector(sel);
const side = $("#side");
const list = $("#list");
const info = $("#info");

const fmtLat = (v) => `${Math.abs(v).toFixed(2)}° ${v < 0 ? "S" : "N"}`;
const fmtLon = (v) => `${Math.abs(v).toFixed(2)}° ${v < 0 ? "W" : "E"}`;

const isLandscape = () => innerWidth > 760 && innerWidth >= innerHeight;

// The part of the screen the planet may use: between the bars, right of the panel
// (above it in portrait).
function freeArea() {
  const top = $(".topbar").getBoundingClientRect().bottom;
  const bottom = $(".statusbar").getBoundingClientRect().top;
  const r = side.getBoundingClientRect();
  if (isLandscape()) return { left: r.right, top, right: innerWidth, bottom };
  return { left: 0, top, right: innerWidth, bottom: r.top };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function locationItem(loc) {
  const li = el("li", "loc");
  li.dataset.id = loc.id;
  const head = el("button", "loc-head");
  head.setAttribute("aria-expanded", "false");
  head.append(el("span", "loc-name", loc.name), el("span", "loc-state", "checking…"),
              el("span", "loc-place", loc.place));
  li.append(head);
  return li;
}

// The 3D map of a location exists once it has been built (scripts/view3d.py).
async function hasViewer(loc) {
  try {
    return (await fetch(loc.viewer, { method: "HEAD" })).ok;
  } catch {
    return false;                                  // not served
  }
}

function showInfo(loc) {
  info.replaceChildren();
  if (!loc) {
    info.append(el("p", "note", "Select a location in the list or on the globe."));
    return;
  }
  const table = el("table");
  for (const [k, v] of [["Place", loc.place], ["Latitude", fmtLat(loc.lat)], ["Longitude", fmtLon(loc.lon)]]) {
    const tr = el("tr");
    tr.append(el("td", "", k), el("td", "", v));
    table.append(tr);
  }
  info.append(table, el("p", "", loc.summary), el("p", "note", loc.detail));
  if (loc.ready) {
    const a = el("a", "btn", "Open 3D map");
    a.href = loc.viewer;
    info.append(a);
  } else {
    const p = el("p", "note", "Build it first: ");
    p.append(el("code", "", `python scripts/view3d.py data/${loc.id}`));
    info.append(el("span", "btn off", loc.ready === false ? "3D map is not built yet" : "Checking for the 3D map…"), p);
  }
}

async function main() {
  const { locations, features = [] } = await (await fetch("locations.json")).json();
  locations.forEach((loc) => list.append(locationItem(loc)));

  let selected = null;
  const byId = (id) => locations.find((l) => l.id === id) ?? null;
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
    showInfo(byId(id));
    if (id) globe.focus(id);
    else globe.release();
  }
  showInfo(null);

  locations.forEach(async (loc) => {
    loc.ready = await hasViewer(loc);
    const state = list.querySelector(`[data-id="${loc.id}"] .loc-state`);
    state.textContent = loc.ready ? "map ready" : "no map yet";
    state.classList.toggle("ready", loc.ready);
    if (selected === loc.id) showInfo(loc);
  });

  list.addEventListener("click", (e) => {
    const head = e.target.closest(".loc-head");
    if (head) select(head.parentElement.dataset.id === selected ? null : head.parentElement.dataset.id);
  });
  addEventListener("keydown", (e) => e.key === "Escape" && select(null));
  addEventListener("globe-cursor", (e) => {
    $("#cursor").textContent = e.detail ? `${fmtLat(e.detail.lat)}, ${fmtLon(e.detail.lon)}` : "";
  });

  // The panel changes size when a place opens (in portrait that moves the planet).
  new ResizeObserver(() => globe.resize()).observe(side);
  window.__gui = { globe, select };
}

main();
