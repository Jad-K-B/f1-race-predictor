import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUpRight,
  Check,
  Download,
  ExternalLink,
  Info,
  Pause,
  Play,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  X,
  AlertTriangle,
  Clock,
  GitBranch,
  CheckCheck,
} from "lucide-react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import MethodPicker from "./MethodPicker";
import PredictedPodium from "./PredictedPodium";
import {
  data,
  labels,
  tasks,
  stages,
  fullName,
  percent,
  colors,
  download,
  type Task,
  type Stage,
  type Method,
} from "./data";
const CarScene = lazy(() => import("./CarScene"));
gsap.registerPlugin(ScrollTrigger);
type View = "race" | "lab" | "archive";
type Modal = "readiness" | "provenance" | "methodology" | "credits" | null;
const taskKeys = Object.keys(tasks) as Task[];
const benchmarkNames: Record<string, string> = {
  race_winner: "Winner",
  podium_finish: "Podium",
  points_finish: "Points",
  order: "Finishing order",
};
const metricNames: Record<string, string> = {
  log_loss: "Log loss",
  brier: "Brier score",
  ece_10: "Calibration error",
  mae: "Rank MAE",
  spearman: "Spearman",
  winner_top1: "Winner top-1",
};

export default function App() {
  const [view, setView] = useState<View>(
    location.hash === "#models"
      ? "lab"
      : location.hash === "#archive"
        ? "archive"
        : "race",
  );
  const [motion, setMotion] = useState(
    !matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [modal, setModal] = useState<Modal>(null),
    dialog = useRef<HTMLDialogElement>(null);
  const [method, setMethod] = useState("external"),
    [task, setTask] = useState<Task>("p_race_winner"),
    [stage, setStage] = useState<Stage>("reconciled");
  const [selectedChoice, setSelected] = useState(""),
    [compareChoice, setCompare] = useState("");
  const [query, setQuery] = useState(""),
    [sort, setSort] = useState("order"),
    [expanded, setExpanded] = useState(false),
    [event, setEvent] = useState(data.events.at(-1)?.id || "upcoming");
  const [scope, setScope] = useState("overall"),
    [benchmarkTask, setBenchmarkTask] = useState("race_winner"),
    [metric, setMetric] = useState("log_loss"),
    [calibration, setCalibration] = useState("external_selected");
  const content = useRef<HTMLDivElement>(null);
  const archiveEntry = data.events.find((item) => item.id === event);
  const forecast = archiveEntry?.forecast;
  const historical = forecast?.evidence_mode === "approximate_retrospective";
  const publication = archiveEntry?.receipt;
  const methods: Record<string, Method> = forecast
    ? { ...forecast.models, ...forecast.baselines }
    : {};
  const teams = Object.fromEntries(
    (forecast?.entrants || []).map((e) => [e.driver_id, e.constructor_id]),
  );
  const current = methods[method] || { drivers: [], predicted_order: [] },
    hasProbability = !!current.drivers[0]?.reconciled;
  const validStages = stages.filter((s) => !!current.drivers[0]?.[s]),
    effectiveStage = validStages.includes(stage)
      ? stage
      : validStages.at(-1) || "reconciled";
  const row =
      current.drivers.find((r) => r.driver_id === selectedChoice) ||
      current.drivers.find((r) => r.predicted_order === 1),
    rival =
      current.drivers.find((r) => r.driver_id === compareChoice) ||
      current.drivers.find((r) => r.predicted_order === 2) ||
      row;
  const selected = row?.driver_id || "",
    compare = rival?.driver_id || "";
  const probability = (driver: typeof row, target = task) =>
    driver?.[effectiveStage]?.[target];
  const favorite = hasProbability
    ? [...current.drivers].sort(
        (a, b) =>
          (b.reconciled?.p_race_winner || 0) -
          (a.reconciled?.p_race_winner || 0),
      )[0].driver_id
    : null;
  const filtered = useMemo(
    () =>
      current.drivers
        .filter((r) =>
          `${fullName(r.driver_id)} ${teams[r.driver_id]}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .sort((a, b) =>
          sort === "probability"
            ? (b[effectiveStage]?.[task] || 0) -
              (a[effectiveStage]?.[task] || 0)
            : a.predicted_order - b.predicted_order,
        ),
    [current, query, sort, effectiveStage, task, event],
  );
  useEffect(() => {
    setSelected("");
    setCompare("");
    setQuery("");
    setExpanded(false);
    setSort("order");
  }, [event]);
  const visible = expanded || query ? filtered : filtered.slice(0, 8);
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setMotion(!media.matches);
    media.addEventListener("change", change);
    return () => media.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.motion = String(motion);
  }, [motion]);
  useEffect(() => {
    if (!motion) return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".hero-heading h1",
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 1, ease: "power4.out" },
      );
      gsap.fromTo(
        ".hero-copy",
        { opacity: 0, y: 18 },
        { opacity: 1, y: 0, duration: 0.8, delay: 0.25 },
      );
      gsap.to(".hero-heading", {
        y: -55,
        opacity: 0,
        ease: "none",
        scrollTrigger: {
          trigger: "#cinema",
          start: "top top",
          end: "65% top",
          scrub: 1,
        },
      });
    });
    return () => ctx.revert();
  }, [motion]);
  useEffect(() => {
    if (modal) dialog.current?.showModal();
    else dialog.current?.close();
  }, [modal]);
  useEffect(() => {
    if (!motion) return;
    const ctx = gsap.context(() => {
      const bars = content.current?.querySelectorAll(".bar-fill");
      const headings = content.current?.querySelectorAll(".view-heading");
      if (bars?.length)
        gsap.fromTo(
          bars,
          { scaleX: 0 },
          { scaleX: 1, duration: 0.6, stagger: 0.025, ease: "power3.out" },
        );
      if (headings?.length)
        gsap.fromTo(
          headings,
          { opacity: 0.3, y: 10 },
          { opacity: 1, y: 0, duration: 0.4 },
        );
    }, content);
    return () => ctx.revert();
  }, [
    view,
    method,
    task,
    effectiveStage,
    motion,
    metric,
    scope,
    benchmarkTask,
  ]);
  function navigate(next: View) {
    setView(next);
    history.replaceState(
      null,
      "",
      next === "race" ? "#race-room" : next === "lab" ? "#models" : "#archive",
    );
    document
      .querySelector("#workspace")
      ?.scrollIntoView({ behavior: motion ? "smooth" : "instant" });
  }
  const benchmarkRows = (benchmarkTask === "order"
    ? data.benchmark.ranking
    : data.benchmark.probability
  ).filter(
    (r) =>
      r.scope === scope && (!("target" in r) || r.target === benchmarkTask),
  ) as unknown as Record<string, string>[];
  const calibrationRows = data.benchmark.calibration.filter(
    (r) =>
      r.scope === scope &&
      r.method === calibration &&
      r.target === benchmarkTask &&
      Number(r.count) > 0,
  );
  const maxMetric = Math.max(
    ...benchmarkRows.map((r) => Number(r[metric])),
    0.01,
  );
  const counts = scope === "overall" ? 48 : 24;
  const interval = (id: string) =>
    data.benchmark.uncertainty.find(
      (r) =>
        r.scope === scope &&
        r.method === id &&
        r.metric === metric &&
        (r.prediction_task === benchmarkTask ||
          (benchmarkTask === "order" && r.prediction_task === "finish_order")),
    );
  return (
    <>
      <a className="skip-link" href="#workspace">
        Skip to race forecasts
      </a>
      <header className="site-header">
        <a className="wordmark" href="#" aria-label="FORMATION home">
          <span className="brand-mark">F/</span> FORMATION
          <span className="wordmark-index">01</span>
        </a>
        <nav aria-label="Main navigation">
          <button onClick={() => navigate("race")}>Race Forecast</button>
          <button onClick={() => navigate("lab")}>Model lab</button>
          <button onClick={() => navigate("archive")}>Forecast archive</button>
        </nav>
        <div className="header-end">
          <span className="mono private-label">
            F1 / MACHINE LEARNING PROJECT
          </span>
          <button
            className="icon-button motion-toggle"
            onClick={() => setMotion(!motion)}
            aria-label={motion ? "Pause motion" : "Enable motion"}
            title={motion ? "Pause motion" : "Enable motion"}
          >
            {motion ? <Pause size={16} /> : <Play size={16} />}
          </button>
        </div>
      </header>
      <main>
        <section id="cinema" className="cinema" aria-labelledby="hero-title">
          <Suspense
            fallback={
              <picture>
                <source
                  media="(max-width: 700px)"
                  srcSet="/assets/car-fallback-mobile.png"
                />
                <img
                  className="loading-poster"
                  src="/assets/car-fallback.png"
                  alt="FORMATION racing concept"
                />
              </picture>
            }
          >
            <CarScene motion={motion} />
          </Suspense>
          <div className="hero-heading">
            <div className="eyebrow">
              <span className="status-dot" /> F1 / MACHINE LEARNING FORECASTING
            </div>
            <h1 id="hero-title">
              FORMATION<span className="hero-slash">/</span>
            </h1>
          </div>
          <div className="hero-copy">
            <p>
              <strong>Formula 1 Race Forecasts</strong>
              <span className="hero-description">
                Machine learning forecasts using historical and current race
                weekend data.
              </span>
            </p>
            <button className="acid-button" onClick={() => navigate("race")}>
              Explore forecasts <ArrowUpRight size={20} />
            </button>
          </div>
          <div className="hero-index mono">
            <span>01 — 03</span>
            <div />
            <ArrowDown size={16} />
          </div>
        </section>
        <div className="release-strip">
          <div>
            <span className="status-dot dark" />
            <strong>
              {data.readiness.forecast_published
                ? "Forecast published"
                : "Next forecast pending"}
            </strong>
            <span>
              {data.readiness.forecast_published
                ? "View the archived forecast and publication record."
                : "Waiting for qualifying results and the confirmed starting grid."}
            </span>
          </div>
          <button onClick={() => setModal("readiness")}>
            Forecast status <ArrowUpRight size={17} />
          </button>
        </div>
        <section id="workspace" className="workspace" ref={content}>
          <div className="workspace-top">
            <span className="eyebrow">
              <span className="section-index">02 /</span> RACE FORECASTS
            </span>
            <span className="mono local-label">
              <ShieldCheck size={14} />{" "}
              {data.publicBuild ? "FORECAST ARCHIVE" : "LOCAL ARCHIVE"} / READ
              ONLY
            </span>
          </div>
          <div
            className="workspace-nav"
            role="tablist"
            aria-label="Forecast workspace"
          >
            {(["race", "lab", "archive"] as View[]).map((v, i) => (
              <button
                key={v}
                role="tab"
                aria-selected={view === v}
                tabIndex={view === v ? 0 : -1}
                id={`tab-${v}`}
                aria-controls={`panel-${v}`}
                onKeyDown={(event) => {
                  const list: View[] = ["race", "lab", "archive"];
                  const next =
                    event.key === "ArrowRight"
                      ? (i + 1) % 3
                      : event.key === "ArrowLeft"
                        ? (i + 2) % 3
                        : event.key === "Home"
                          ? 0
                          : event.key === "End"
                            ? 2
                            : -1;
                  if (next < 0) return;
                  event.preventDefault();
                  navigate(list[next]);
                  document.getElementById(`tab-${list[next]}`)?.focus();
                }}
                onClick={() => navigate(v)}
              >
                <span className="mono">0{i + 1}</span>
                {v === "race"
                  ? "Race Forecast"
                  : v === "lab"
                    ? "Model lab"
                    : "Forecast archive"}
                <ArrowUpRight size={16} />
              </button>
            ))}
          </div>
          {view === "race" && (
            <div role="tabpanel" id="panel-race" aria-labelledby="tab-race">
              <div className="view-heading">
                <div>
                  <p className="eyebrow">
                    {forecast
                      ? `ROUND ${String(forecast.race.round).padStart(2, "0")} / ${forecast.race.year} / ${fullName(forecast.race.circuit_id)}`
                      : "UPCOMING RACE"}
                  </p>
                  <h2>
                    {forecast
                      ? `${fullName(forecast.race.grand_prix_id)} Grand Prix`
                      : "The next forecast"}
                    <span className="accent-dot">.</span>
                  </h2>
                </div>
                <label className="select-label">
                  Forecast archive
                  <select
                    aria-label="Forecast archive"
                    value={event}
                    onChange={(e) => setEvent(e.target.value)}
                  >
                    {data.events.map((item) => (
                      <option key={item.id} value={item.id}>
                        {fullName(item.forecast.race.grand_prix_id)}{" "}
                        {item.forecast.race.year} ·{" "}
                        {item.forecast.evidence_mode ===
                        "approximate_retrospective"
                          ? "Replay"
                          : item.receipt
                            ? "Published forecast"
                            : "Unpublished forecast"}
                      </option>
                    ))}
                    <option value="upcoming">Upcoming · Inputs pending</option>
                  </select>
                </label>
              </div>
              {!forecast || !row ? (
                <div className="pending-view">
                  <Clock size={36} />
                  <div>
                    <p className="eyebrow">FORECAST NOT AVAILABLE YET</p>
                    <h3>Waiting for race weekend inputs.</h3>
                    <p>
                      A genuine pre-race forecast is published only after the
                      entries, driver eligibility, qualifying results and
                      confirmed starting grid have been reviewed. No starting
                      positions or probabilities are filled in early.
                    </p>
                    <button
                      className="dark-button"
                      onClick={() => setModal("readiness")}
                    >
                      View input status <ArrowUpRight size={17} />
                    </button>
                    <small>
                      Status snapshot:{" "}
                      {new Date(data.readiness.recorded_at).toUTCString()}. Not
                      a live feed.
                    </small>
                  </div>
                </div>
              ) : (
                <>
                  <div className="context-note">
                    <Info size={15} />
                    <p>
                      {historical ? (
                        <>
                          <strong>Historical replay.</strong> These saved
                          estimates were generated after the race. They were not
                          published before the event or used as independent
                          validation, and the 2023 calibrators were fitted using
                          that season.
                        </>
                      ) : (
                        <>
                          <strong>
                            {publication
                              ? "Published pre-race forecast."
                              : "Unpublished forecast."}
                          </strong>{" "}
                          Cutoff: {new Date(forecast.cutoff).toUTCString()}.{" "}
                          {publication
                            ? `Publication verified: ${new Date(publication.published_at).toUTCString()}.`
                            : "No verified public receipt is available. This is not proof of pre-race publication."}
                        </>
                      )}
                    </p>
                    <button
                      onClick={() => setModal("provenance")}
                      title={
                        historical
                          ? "View historical replay details"
                          : "View forecast details"
                      }
                      aria-label={
                        historical
                          ? "View historical replay details"
                          : "View forecast details"
                      }
                    >
                      <ArrowUpRight size={18} />
                    </button>
                  </div>
                  <div className="race-toolbar">
                    <div className="select-label forecast-method">
                      Method
                      <MethodPicker
                        label="Forecast method"
                        value={method}
                        onChange={(value) => {
                          setMethod(value);
                          if (!methods[value].drivers[0].reconciled)
                            setSort("order");
                        }}
                        options={Object.keys(methods).map((key) => ({
                          value: key,
                          label: labels[key],
                          detail:
                            key === "external" || key === "numpy"
                              ? "Model pipeline"
                              : methods[key].drivers[0].reconciled
                                ? "Probability baseline"
                                : "Order-only baseline",
                        }))}
                      />
                    </div>
                    <div className="segmented" aria-label="Prediction target">
                      {taskKeys.map((t) => (
                        <button
                          key={t}
                          aria-pressed={task === t}
                          disabled={!hasProbability}
                          onClick={() => setTask(t)}
                        >
                          {tasks[t]}
                        </button>
                      ))}
                    </div>
                    <label className="select-label">
                      Probability stage
                      <select
                        aria-label="Probability stage"
                        value={effectiveStage}
                        disabled={!hasProbability}
                        onChange={(e) => setStage(e.target.value as Stage)}
                      >
                        {(validStages.length
                          ? validStages
                          : ["reconciled"]
                        ).map((s) => (
                          <option key={s} value={s}>
                            {fullName(s.replace("_", "-"))}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="icon-button export-button"
                      title="Download selected saved predictions"
                      aria-label="Download selected saved predictions"
                      onClick={() =>
                        download(
                          `formation-${historical ? `replay-${forecast.race.race_id}` : forecast.snapshot_id}-${method}.json`,
                          {
                            notice: forecast.notice,
                            archive: forecast,
                            publication,
                            method,
                            prediction: current,
                          },
                        )
                      }
                    >
                      <Download size={18} />
                    </button>
                  </div>
                  <PredictedPodium
                    season={forecast.race.year}
                    entrants={forecast.entrants}
                    evidenceLabel={
                      historical
                        ? "HISTORICAL REPLAY"
                        : publication
                          ? "PUBLISHED PRE-RACE"
                          : "UNPUBLISHED FORECAST"
                    }
                    prediction={current}
                    methodLabel={labels[method]}
                    stage={effectiveStage}
                    selectedDriver={selected}
                    onSelectDriver={setSelected}
                  />
                  <div className="race-layout">
                    <div className="field-panel">
                      <div className="field-top">
                        <span className="eyebrow">
                          THE PREDICTED FIELD{" "}
                          <span className="muted">
                            / {current.drivers.length}
                          </span>
                        </span>
                        <div className="field-tools">
                          <label className="search-box">
                            <Search size={15} />
                            <input
                              aria-label="Search drivers"
                              placeholder="Find a driver"
                              value={query}
                              onChange={(e) => setQuery(e.target.value)}
                            />
                          </label>
                          <label className="sort-label">
                            <SlidersHorizontal size={16} />
                            <select
                              aria-label="Sort drivers"
                              value={sort}
                              onChange={(e) => setSort(e.target.value)}
                            >
                              <option value="order">Order</option>
                              <option
                                value="probability"
                                disabled={!hasProbability}
                              >
                                Probability
                              </option>
                            </select>
                          </label>
                        </div>
                      </div>
                      <div className="table-scroll">
                        <table className="field-table">
                          <thead>
                            <tr>
                              <th scope="col">POS</th>
                              <th scope="col">DRIVER / TEAM</th>
                              <th scope="col">
                                {hasProbability
                                  ? `${tasks[task].toUpperCase()} %`
                                  : "ORDER ONLY"}
                              </th>
                              <th scope="col" className="secondary-col">
                                PODIUM
                              </th>
                              <th scope="col" className="secondary-col">
                                POINTS
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {visible.map((r) => (
                              <tr
                                key={r.driver_id}
                                className={
                                  selected === r.driver_id ? "selected" : ""
                                }
                              >
                                <td className="position">
                                  {String(r.predicted_order).padStart(2, "0")}
                                </td>
                                <td>
                                  <button
                                    className="driver-button"
                                    onClick={() => setSelected(r.driver_id)}
                                    aria-pressed={selected === r.driver_id}
                                  >
                                    <span
                                      className="team-line"
                                      style={{
                                        background: colors[teams[r.driver_id]],
                                      }}
                                    />
                                    <span>
                                      <strong>{fullName(r.driver_id)}</strong>
                                      <small>
                                        {fullName(teams[r.driver_id])}
                                      </small>
                                    </span>
                                  </button>
                                </td>
                                <td>
                                  <div className="prob-cell">
                                    <span className="bar-track">
                                      <span
                                        className="bar-fill"
                                        style={{
                                          width: `${(probability(r) || 0) * 100}%`,
                                        }}
                                      />
                                    </span>
                                    <span className="mono">
                                      {percent(probability(r))}
                                    </span>
                                  </div>
                                </td>
                                <td className="secondary-col mono">
                                  {percent(probability(r, "p_podium_finish"))}
                                </td>
                                <td className="secondary-col mono">
                                  {percent(probability(r, "p_points_finish"))}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {!visible.length && (
                        <p className="empty">No drivers match “{query}”.</p>
                      )}
                      <div className="table-footer">
                        <span className="mono">
                          {visible.length} / {filtered.length} DRIVERS
                        </span>
                        <button onClick={() => setExpanded(!expanded)}>
                          {expanded ? "Top eight" : "Full field"}
                          <ArrowDown
                            size={15}
                            style={{
                              transform: expanded
                                ? "rotate(180deg)"
                                : undefined,
                            }}
                          />
                        </button>
                      </div>
                    </div>
                    <aside className="driver-focus" aria-label="Driver detail">
                      <div className="focus-heading">
                        <span className="eyebrow">DRIVER FOCUS</span>
                        <span className="mono">
                          P{String(row.predicted_order).padStart(2, "0")}
                        </span>
                      </div>
                      <div
                        className="driver-code"
                        style={{ color: colors[teams[selected]] }}
                      >
                        {selected.split("-").at(-1)?.slice(0, 3)}
                      </div>
                      <h3>{fullName(selected)}</h3>
                      <p className="eyebrow muted">
                        {fullName(teams[selected])}
                      </p>
                      <div className="focus-value">
                        {percent(probability(row))}
                        <span>
                          {hasProbability
                            ? `${tasks[task]} probability`
                            : "Order-only baseline"}
                        </span>
                      </div>
                      <div className="stage-chart">
                        {validStages.map((s) => (
                          <div key={s}>
                            <span>{fullName(s.replace("_", "-"))}</span>
                            <span className="bar-track">
                              <span
                                className="bar-fill"
                                style={{ width: `${row[s]![task] * 100}%` }}
                              />
                            </span>
                            <span className="mono">
                              {percent(row[s]![task])}
                            </span>
                          </div>
                        ))}
                      </div>
                      <label className="select-label compare-label">
                        Compare with
                        <select
                          aria-label="Compare driver"
                          value={compare}
                          onChange={(e) => setCompare(e.target.value)}
                        >
                          {current.drivers.map((d) => (
                            <option key={d.driver_id} value={d.driver_id}>
                              {fullName(d.driver_id)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <div className="head-to-head">
                        <div>
                          <span>{fullName(selected)}</span>
                          <strong>{percent(probability(row))}</strong>
                        </div>
                        <span className="mono">VS</span>
                        <div>
                          <span>{fullName(compare)}</span>
                          <strong>{percent(probability(rival))}</strong>
                        </div>
                      </div>
                      <p className="micro-note">
                        Separate marginal estimates, not a head-to-head or exact
                        podium-order probability.
                      </p>
                    </aside>
                  </div>
                  <div className="consistency">
                    <div>
                      {favorite && favorite !== current.predicted_order[0] ? (
                        <AlertTriangle size={20} />
                      ) : (
                        <CheckCheck size={20} />
                      )}
                      <div>
                        <strong>
                          {!favorite
                            ? "Order-only reference"
                            : favorite === current.predicted_order[0]
                              ? "Ranking and winner favorite agree"
                              : "Ranking and winner favorite disagree"}
                        </strong>
                        <p>
                          {favorite
                            ? `Ranking P1: ${fullName(current.predicted_order[0])}. Highest reconciled winner probability: ${fullName(favorite)}.`
                            : "This baseline does not supply probabilities. No marginal estimates are substituted."}
                        </p>
                      </div>
                    </div>
                    <div className="quota">
                      <span className="eyebrow">
                        {effectiveStage.toUpperCase()} FIELD TOTALS
                      </span>
                      <strong className="mono">
                        {hasProbability
                          ? taskKeys
                              .map((t) =>
                                current.drivers
                                  .reduce(
                                    (s, d) => s + (d[effectiveStage]?.[t] || 0),
                                    0,
                                  )
                                  .toFixed(2),
                              )
                              .join(" / ")
                          : "— / — / —"}
                      </strong>
                      <small>
                        Winner / podium / points · reconciled quotas 1 / 3 / 10
                      </small>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
          {view === "lab" && (
            <div
              role="tabpanel"
              id="panel-lab"
              aria-labelledby="tab-lab"
              className="model-lab"
            >
              <div className="view-heading">
                <div>
                  <p className="eyebrow">
                    HISTORICAL MODEL RESULTS / COMPLETED TEST
                  </p>
                  <h2>
                    Model comparisons<span className="accent-dot">.</span>
                  </h2>
                </div>
                <button
                  className="outline-button"
                  onClick={() => setModal("methodology")}
                >
                  Methodology <ArrowUpRight size={17} />
                </button>
              </div>
              {data.benchmark.probability.length === 0 ? (
                <p className="pending-view">
                  Historical evaluation reports are not included in this public
                  forecast release.
                </p>
              ) : (
                <>
                  <div className="context-note">
                    <ShieldCheck size={16} />
                    <p>
                      <strong>Historical test results: 2024–2025.</strong> These
                      are saved results from the completed final evaluation, not
                      live forecasts or validation results. This page does not
                      select or evaluate models.
                    </p>
                  </div>
                  <div className="lab-controls">
                    <div className="segmented">
                      {["overall", "2024", "2025"].map((s) => (
                        <button
                          key={s}
                          aria-pressed={scope === s}
                          onClick={() => setScope(s)}
                        >
                          {s === "overall" ? "Both seasons" : s}
                        </button>
                      ))}
                    </div>
                    <label className="select-label">
                      Prediction task
                      <select
                        aria-label="Benchmark task"
                        value={benchmarkTask}
                        onChange={(e) => {
                          setBenchmarkTask(e.target.value);
                          setMetric(
                            e.target.value === "order" ? "mae" : "log_loss",
                          );
                        }}
                      >
                        {Object.entries(benchmarkNames).map(([k, v]) => (
                          <option key={k} value={k}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="select-label">
                      Metric
                      <select
                        aria-label="Benchmark metric"
                        value={metric}
                        onChange={(e) => setMetric(e.target.value)}
                      >
                        {(benchmarkTask === "order"
                          ? ["mae", "spearman", "winner_top1"]
                          : ["log_loss", "brier", "ece_10"]
                        ).map((m) => (
                          <option key={m} value={m}>
                            {metricNames[m]}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="lab-layout">
                    <section className="benchmark-chart">
                      <div className="chart-heading">
                        <h3>{metricNames[metric]}</h3>
                        <span className="mono">
                          {counts} RACES /{" "}
                          {["spearman", "winner_top1"].includes(metric)
                            ? "HIGHER"
                            : "LOWER"}{" "}
                          IS BETTER
                        </span>
                      </div>
                      {benchmarkRows.map((r) => {
                        const ci = interval(r.method);
                        return (
                          <div className="benchmark-row" key={r.method}>
                            <div>
                              <strong>{labels[r.method] || r.method}</strong>
                              <span className="mono">
                                {Number(r[metric]).toFixed(4)}
                              </span>
                            </div>
                            <div className="benchmark-track">
                              <div
                                className={`bar-fill ${r.method.includes("external") ? "external" : r.method.includes("numpy") ? "numpy" : "baseline"}`}
                                style={{
                                  width: `${Math.max(0, (Number(r[metric]) / maxMetric) * 100)}%`,
                                }}
                              />
                            </div>
                            {ci && (
                              <small className="mono">
                                RACE-MEAN 95% CI{" "}
                                {Number(ci.lower_95).toFixed(3)} —{" "}
                                {Number(ci.upper_95).toFixed(3)}
                              </small>
                            )}
                          </div>
                        );
                      })}
                      <p className="micro-note">
                        All predetermined methods remain visible. Aggregate
                        metrics and race-mean bootstrap intervals use different
                        weighting; intervals are not confidence bounds for the
                        aggregate estimate.
                      </p>
                    </section>
                    <section className="calibration-panel">
                      <div className="chart-heading">
                        <h3>
                          {benchmarkTask === "order"
                            ? "Order performance"
                            : "Calibration"}
                        </h3>
                        {benchmarkTask !== "order" && (
                          <MethodPicker
                            label="Calibration method"
                            value={calibration}
                            onChange={setCalibration}
                            options={[
                              {
                                value: "external_selected",
                                label: "External ML",
                                detail: "Model pipeline",
                              },
                              {
                                value: "numpy_selected",
                                label: "NumPy",
                                detail: "Model pipeline",
                              },
                            ]}
                          />
                        )}
                      </div>
                      {benchmarkTask === "order" ? (
                        <div className="ranking-stats">
                          {benchmarkRows
                            .filter((r) => r.method.includes("selected"))
                            .map((r) => (
                              <div key={r.method}>
                                <p className="eyebrow">{labels[r.method]}</p>
                                <strong>
                                  {percent(Number(r.podium_order_exact))}
                                </strong>
                                <span>Exact P1–P3 order</span>
                                <p>
                                  {percent(Number(r.winner_top1))} winner top-1
                                </p>
                                <p>
                                  {Number(r.spearman).toFixed(3)} Spearman
                                  correlation
                                </p>
                              </div>
                            ))}
                        </div>
                      ) : (
                        <>
                          <svg
                            className="calibration-chart"
                            viewBox="0 0 350 310"
                            role="img"
                            aria-label={`${labels[calibration]} ${benchmarkNames[benchmarkTask]} calibration: predicted probability versus observed frequency`}
                          >
                            <path
                              d="M42 20V267H323"
                              stroke="currentColor"
                              fill="none"
                            />
                            {[0, 0.25, 0.5, 0.75, 1].map((n) => (
                              <g key={n}>
                                <path
                                  d={`M42 ${267 - n * 240}H323`}
                                  stroke="currentColor"
                                  opacity=".12"
                                />
                                <text x="33" y={271 - n * 240} textAnchor="end">
                                  {n}
                                </text>
                                <text
                                  x={42 + n * 280}
                                  y="285"
                                  textAnchor="middle"
                                >
                                  {n}
                                </text>
                              </g>
                            ))}
                            <path
                              d="M42 267L322 27"
                              stroke="#7a858c"
                              strokeDasharray="4 5"
                            />
                            <polyline
                              points={calibrationRows
                                .map(
                                  (r) =>
                                    `${42 + Number(r.mean_prediction) * 280},${267 - Number(r.observed_rate) * 240}`,
                                )
                                .join(" ")}
                              fill="none"
                              stroke="#8ea717"
                              strokeWidth="2"
                            />
                            {calibrationRows.map((r) => (
                              <circle
                                tabIndex={0}
                                key={r.bin}
                                cx={42 + Number(r.mean_prediction) * 280}
                                cy={267 - Number(r.observed_rate) * 240}
                                r={
                                  4 +
                                  Math.min(6, Math.sqrt(Number(r.count)) / 5)
                                }
                                fill="#c8e83a"
                                stroke="#252c21"
                              >
                                <title>{`Predicted ${percent(Number(r.mean_prediction))}; observed ${percent(Number(r.observed_rate))}; n=${r.count}`}</title>
                              </circle>
                            ))}
                            <text x="180" y="306" textAnchor="middle">
                              PREDICTED PROBABILITY
                            </text>
                            <text
                              x="13"
                              y="155"
                              textAnchor="middle"
                              transform="rotate(-90 13 155)"
                            >
                              OBSERVED FREQUENCY
                            </text>
                          </svg>
                          <p className="micro-note">
                            Dashed line: perfect calibration. Circle size
                            reflects bin sample count. Sparse bins are
                            uncertain, particularly for race winners.
                          </p>
                        </>
                      )}
                    </section>
                  </div>
                  <div className="lab-footer">
                    <p>
                      <strong>Compare every method in context.</strong> A more
                      complex model is not automatically more accurate.
                      Baselines, calibration and changes between seasons still
                      matter.
                    </p>
                    <button
                      className="outline-button"
                      onClick={() =>
                        download(
                          `formation-final-test-${scope}-${benchmarkTask}.json`,
                          {
                            notice:
                              "Previously completed final-test report, not validation. No new evaluation.",
                            scope,
                            task: benchmarkTask,
                            rows: benchmarkRows,
                            uncertainty: data.benchmark.uncertainty.filter(
                              (r) => r.scope === scope,
                            ),
                          },
                        )
                      }
                    >
                      <Download size={16} /> Export report
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
          {view === "archive" && (
            <div
              role="tabpanel"
              id="panel-archive"
              aria-labelledby="tab-archive"
            >
              <div className="view-heading">
                <div>
                  <p className="eyebrow">
                    HISTORICAL FORECASTS / SAVED RECORDS
                  </p>
                  <h2>
                    Forecast history<span className="accent-dot">.</span>
                  </h2>
                </div>
                <span className="mono">
                  {
                    data.events.filter(
                      (item) =>
                        item.forecast.evidence_mode ===
                        "approximate_retrospective",
                    ).length
                  }{" "}
                  REPLAY / {data.events.filter((item) => item.receipt).length}{" "}
                  PRE-RACE
                </span>
              </div>
              {data.events.length === 0 && (
                <p className="pending-view">
                  No verified forecast has been published yet.
                </p>
              )}
              {data.events.map((item) => (
                <div className="archive-row" key={item.id}>
                  <span className="archive-year">
                    {item.forecast.race.year}
                  </span>
                  <div>
                    <p className="eyebrow">
                      ROUND {item.forecast.race.round} ·{" "}
                      {fullName(item.forecast.race.grand_prix_id)}
                    </p>
                    <h3>
                      {fullName(item.forecast.race.circuit_id)},{" "}
                      {item.forecast.evidence_mode ===
                      "approximate_retrospective"
                        ? "historical replay"
                        : item.receipt
                          ? "published forecast"
                          : "unpublished forecast"}
                      .
                    </h3>
                    <p>
                      {item.forecast.entrants.length} drivers. Two model
                      pipelines. Six fixed baselines.
                    </p>
                    <span className="badge">
                      {item.forecast.evidence_mode ===
                      "approximate_retrospective"
                        ? "HISTORICAL REPLAY / NOT PUBLISHED PRE-RACE"
                        : item.receipt
                          ? "VERIFIED PRE-RACE PUBLICATION"
                          : "NO VERIFIED PUBLICATION"}
                    </span>
                  </div>
                  <button
                    className="dark-button"
                    onClick={() => {
                      setEvent(item.id);
                      navigate("race");
                    }}
                  >
                    Open race <ArrowUpRight size={17} />
                  </button>
                </div>
              ))}
              <div className="archive-facts">
                <div>
                  <GitBranch size={20} />
                  <h3>Fixed model version</h3>
                  <p>
                    Model families, features, calibration and reconciliation
                    remain unchanged.
                  </p>
                  <code>
                    {forecast?.release_manifest_sha256.slice(0, 20) ||
                      "Frozen v2"}
                    …
                  </code>
                </div>
                <div>
                  <Clock size={20} />
                  <h3>Timing matters</h3>
                  <p>
                    A historical cutoff alone does not prove that a forecast was
                    published before the start. Published records require an
                    independently checked receipt and matching public bytes.
                  </p>
                </div>
                <div>
                  <ShieldCheck size={20} />
                  <h3>Genuine pre-race forecasts</h3>
                  <p>
                    Upcoming-race forecasts are published only after the inputs
                    are reviewed. An independent server receipt records when the
                    forecast was made public.
                  </p>
                  <a
                    href="https://github.com/Jad-K-B/f1-race-forecasts"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Public forecast repository <ExternalLink size={14} />
                  </a>
                </div>
              </div>
              <button
                className="outline-button"
                onClick={() => setModal("provenance")}
              >
                View archive details <ArrowUpRight size={16} />
              </button>
            </div>
          )}
        </section>
        <section className="method-band">
          <div>
            <span className="eyebrow">03 / HOW FORECASTS ARE PRODUCED</span>
            <h2>
              Models, data,
              <br />
              and clear limits.
            </h2>
          </div>
          <div className="method-principles">
            <p>
              <span>01</span>Models fixed before final testing.
              <Check size={17} />
            </p>
            <p>
              <span>02</span>Baselines shown for comparison.
              <Check size={17} />
            </p>
            <p>
              <span>03</span>Probabilities are not guarantees.
              <Check size={17} />
            </p>
            <button onClick={() => setModal("methodology")}>
              Read the methodology <ArrowUpRight size={17} />
            </button>
          </div>
        </section>
      </main>
      <footer>
        <a className="wordmark" href="#">
          <span className="brand-mark">F/</span>FORMATION
        </a>
        <p>Not affiliated with Formula 1, the FIA or any team.</p>
        <button onClick={() => setModal("credits")}>
          Credits & data rights <ArrowUpRight size={14} />
        </button>
        <span className="mono">FORMATION / 2026</span>
      </footer>
      <dialog
        ref={dialog}
        onCancel={() => setModal(null)}
        onClick={(e) => {
          if (e.target === dialog.current) setModal(null);
        }}
      >
        <div className="dialog-top">
          <span className="eyebrow">FORMATION / RECORD</span>
          <button
            className="icon-button"
            aria-label="Close dialog"
            onClick={() => setModal(null)}
          >
            <X size={20} />
          </button>
        </div>
        {modal === "readiness" && (
          <>
            <h2>Forecast input status.</h2>
            <p className="badge">WAITING FOR RACE WEEKEND INPUTS</p>
            <p>
              Saved {new Date(data.readiness.recorded_at).toUTCString()}. This
              is an archived status, not a current schedule or live data check.
            </p>
            <ol className="blocker-list">
              {data.readiness.blockers.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ol>
            <p>
              {data.readiness.forecast_published
                ? "Verified publication records are available in the forecast archive."
                : data.events.some(
                      (item) => item.forecast.evidence_mode === "prospective",
                    )
                  ? "An unpublished forecast is available for local inspection; no verified public receipt is recorded."
                  : "No genuine pre-race forecast or pre-weekend snapshot exists in this saved frontend data."}{" "}
              Race-weekend data collection and model execution remain separate
              from this interface.
            </p>
          </>
        )}
        {modal === "provenance" && (
          <>
            <h2>
              {historical ? "Historical replay details." : "Forecast details."}
            </h2>
            <dl className="provenance-list">
              {Object.entries(
                forecast
                  ? {
                      snapshot_id: forecast.snapshot_id,
                      evidence_mode: forecast.evidence_mode,
                      cutoff: forecast.cutoff,
                      generated_at: forecast.generated_at,
                      completed_at: forecast.completed_at,
                      race_start: forecast.race.race_start || "Unavailable",
                      git_commit: forecast.git_commit,
                      release_manifest_sha256: forecast.release_manifest_sha256,
                      archive_sha256: forecast.archive_sha256,
                      published_at:
                        publication?.published_at || "No verified publication",
                    }
                  : {},
              ).map(([k, v]) => (
                <div key={k}>
                  <dt>{fullName(k.replaceAll("_", "-"))}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
            <p>
              {historical
                ? "No record of a pre-race publication is available for this replay. Its historical cutoff is approximate and does not prove what was known at that exact time."
                : publication
                  ? "Public forecast bytes match the recorded server receipt."
                  : "No independently verified publication receipt is available."}
            </p>
            {publication && (
              <a href={publication.asset_url} target="_blank" rel="noreferrer">
                Published forecast <ExternalLink size={14} />
              </a>
            )}
            <details>
              <summary>
                Source artifact hashes / {forecast?.sources.length || 0}
              </summary>
              {forecast?.sources.map((source, i) => (
                <div className="hash-row" key={i}>
                  <strong>{source.url}</strong>
                  <code>{source.sha256}</code>
                </div>
              ))}
            </details>
          </>
        )}
        {modal === "methodology" && (
          <>
            <h2>How the forecasts work.</h2>
            <div className="methodology-copy">
              <h3>Two fixed model pipelines</h3>
              <p>
                External ML uses logistic regression for winner probability,
                XGBoost for podium probability, Random Forest for points
                probability and XGBoost ranking for finishing order. The NumPy
                pipeline and six predetermined references are retained alongside
                it.
              </p>
              <h3>Training and testing follow time</h3>
              <p>
                Training: 2014–2022. Validation: 2023. Final testing: 2024–2025.
                Preprocessing is training-only. The interface reads saved
                reports and never fits or evaluates a model.
              </p>
              <h3>Three probability targets</h3>
              <p>
                Winner, podium and points are separate targets. Reconciliation
                imposes nested probabilities and field quotas of 1, 3 and 10. A
                predicted P1–P3 order is not an exact-order probability
                distribution.
              </p>
              <h3>Upcoming forecasts need reviewed inputs</h3>
              <p>
                The Bahrain replay is historical, not proof of a forecast made
                before the race or an independent validation result. A genuine
                upcoming-race forecast requires reviewed entries, driver
                eligibility, qualifying results and the confirmed starting grid.
                Missing information blocks publication; it is never invented.
              </p>
            </div>
          </>
        )}
        {modal === "credits" && (
          <>
            <h2>Made with attribution.</h2>
            <p>
              3D geometry:{" "}
              <a
                href="https://sketchfab.com/3d-models/f1-2026-concept-polygon-model-ea3bde709b1e4dc9b0ec8557d106ed42"
                target="_blank"
                rel="noreferrer"
              >
                F1 2026 concept by Qvist_designs
              </a>
              ,{" "}
              <a
                href="https://creativecommons.org/licenses/by/4.0/"
                target="_blank"
                rel="noreferrer"
              >
                CC BY 4.0
              </a>
              . Assembly preparation by filan214. FORMATION adds geometry
              simplification, compression, custom materials, lighting and camera
              choreography. This is an independent concept, not official team
              CAD.
            </p>
            <p>
              Three.js, React, GSAP and Lucide. Barlow Condensed and IBM Plex
              Mono under SIL Open Font License. The fallback is a render of the
              same licensed racing concept.
            </p>
            <p>
              {!data.publicBuild && (
                <>
                  <strong>Local application only.</strong> The saved replay
                  fixture is not cleared for public redistribution.{" "}
                </>
              )}
              Raw FIA/F1 documents, feature matrices, trained models and
              credentials are not served by this application. The forecast-only
              public repository is a separate destination.
            </p>
          </>
        )}
      </dialog>
    </>
  );
}
