"use client";

import { useEffect, useState, type CSSProperties } from "react";

type Point = { x: number; y: number };
type Order = {
  id: string;
  pickup: Point;
  dropoff: Point;
  releaseTime: number;
  deadline: number;
  priority: number;
};
type Event = {
  time: number;
  kind: "order_arrival" | "cell_blocked" | "cell_unblocked";
  position?: Point;
  orderId?: string;
};
type TraceStep = {
  time: number;
  action: string;
  position: [number, number];
  battery: number;
  reward: number;
  cumulativeReward: number;
  carriedOrderId: string | null;
  deliveredOrderId: string | null;
  violations: string[];
  eventCount: number;
};
type Metrics = {
  weightedCompletionRate: number;
  weightedOnTimeCompletionRate: number;
  totalReward: number;
  completedOrders: number;
  totalOrders: number;
  constraintViolations: number;
  decisionTimeMs: number;
  steps: number;
};
type AgentResult = {
  id: string;
  label: string;
  family: string;
  description: string;
  color: string;
  metrics: Metrics;
  planningCalls: number;
  expandedNodes: number;
  learningUpdates: number;
  trace: TraceStep[];
};
type DemoData = {
  generatedAt: string;
  verificationStatus: string;
  scenarioFingerprint: string;
  scenario: {
    id: string;
    width: number;
    height: number;
    horizon: number;
    batteryCapacity: number;
    initialRobot: Point;
    obstacles: Point[];
    chargingStations: Point[];
    orders: Order[];
    events: Event[];
  };
  agents: AgentResult[];
};

type RouteDirection = "north" | "east" | "south" | "west";
type OrderState = "queued" | "ready" | "picked" | "carried" | "delivered" | "expired";

const actionLabels: Record<string, string> = {
  up: "Move north",
  down: "Move south",
  left: "Move west",
  right: "Move east",
  pickup: "Pick up order",
  dropoff: "Deliver order",
  charge: "Recharge",
  wait: "Wait",
};

const eventLabels: Record<Event["kind"], string> = {
  order_arrival: "Order released",
  cell_blocked: "Aisle closed",
  cell_unblocked: "Aisle reopened",
};

const cellKey = (point: Point) => `${point.x}:${point.y}`;

const rackIslandLabels = new Map([
  ["2:2", "R1"],
  ["2:7", "R2"],
  ["6:2", "R3"],
  ["6:7", "R4"],
  ["10:2", "R5"],
  ["10:7", "R6"],
]);

function blockedCells(events: Event[], time: number) {
  const cells = new Set<string>();
  for (const event of events) {
    if (event.time > time || !event.position) continue;
    const key = cellKey(event.position);
    if (event.kind === "cell_blocked") cells.add(key);
    if (event.kind === "cell_unblocked") cells.delete(key);
  }
  return cells;
}

function buildRoute(points: Point[]) {
  const cells = new Map<string, Set<RouteDirection>>();
  const ensureCell = (point: Point) => {
    const key = cellKey(point);
    if (!cells.has(key)) cells.set(key, new Set<RouteDirection>());
    return cells.get(key)!;
  };

  points.forEach((point) => ensureCell(point));
  for (let index = 1; index < points.length; index += 1) {
    const from = points[index - 1];
    const to = points[index];
    const fromCell = ensureCell(from);
    const toCell = ensureCell(to);

    if (to.x === from.x + 1 && to.y === from.y) {
      fromCell.add("east");
      toCell.add("west");
    } else if (to.x === from.x - 1 && to.y === from.y) {
      fromCell.add("west");
      toCell.add("east");
    } else if (to.y === from.y + 1 && to.x === from.x) {
      fromCell.add("south");
      toCell.add("north");
    } else if (to.y === from.y - 1 && to.x === from.x) {
      fromCell.add("north");
      toCell.add("south");
    }
  }
  return cells;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 1 }).format(value);
}

function pointFromTrace(step: TraceStep): Point {
  return { x: step.position[0], y: step.position[1] };
}

function RouteLayer({
  points,
  width,
  height,
  phase,
}: {
  points: Point[];
  width: number;
  height: number;
  phase: "travelled" | "projected";
}) {
  const route = buildRoute(points);
  return (
    <div
      className={`route-layer route-${phase}`}
      style={{
        gridTemplateColumns: `repeat(${width}, 1fr)`,
        gridTemplateRows: `repeat(${height}, 1fr)`,
      }}
      aria-hidden="true"
    >
      {Array.from({ length: width * height }).map((_, index) => {
        const point = { x: index % width, y: Math.floor(index / width) };
        const directions = route.get(cellKey(point));
        return (
          <span className="route-cell" key={`${phase}-${cellKey(point)}`}>
            {directions?.size ? <i className="route-node" /> : null}
            {directions
              ? Array.from(directions).map((direction) => (
                  <i className={`route-branch route-${direction}`} key={direction} />
                ))
              : null}
          </span>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentId, setAgentId] = useState("hybrid");
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch("./demo-data.json")
      .then((response) => {
        if (!response.ok) throw new Error(`demo artifact returned ${response.status}`);
        return response.json() as Promise<DemoData>;
      })
      .then((payload) => {
        setData(payload);
        const preferred = payload.agents.find((candidate) => candidate.id === "hybrid");
        setAgentId(preferred?.id ?? payload.agents[0]?.id ?? "");
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "could not load demo artifact");
      });
  }, []);

  const agent = data?.agents.find((candidate) => candidate.id === agentId) ?? null;
  const maximumFrame = Math.max((agent?.trace.length ?? 1) - 1, 0);

  const selectAgent = (nextAgentId: string) => {
    setAgentId(nextAgentId);
    setFrame(0);
    setPlaying(false);
  };

  useEffect(() => {
    if (!playing || maximumFrame === 0) return;
    const timer = window.setInterval(() => {
      setFrame((current) => {
        if (current >= maximumFrame) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 560);
    return () => window.clearInterval(timer);
  }, [playing, maximumFrame]);

  const currentStep = agent?.trace[Math.min(frame, maximumFrame)] ?? null;
  const currentTime = currentStep?.time ?? 0;
  const blocked = blockedCells(data?.scenario.events ?? [], currentTime);
  const obstacleSet = new Set((data?.scenario.obstacles ?? []).map(cellKey));
  const chargerSet = new Set((data?.scenario.chargingStations ?? []).map(cellKey));

  if (error) {
    return (
      <main className="loading-shell error-shell">
        <p className="paper-label">Artifact error</p>
        <h1>Replay unavailable</h1>
        <p>{error}</p>
        <a href="https://github.com/Revincxt/adaptive-agent-lab">Open the repository</a>
      </main>
    );
  }

  if (!data || !agent || !currentStep) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-rule" aria-hidden="true" />
        <p className="paper-label">Adaptive Agent Lab</p>
        <h1>Loading experiment artifact…</h1>
      </main>
    );
  }

  const robotPosition = pointFromTrace(currentStep);
  const tracePoints = [data.scenario.initialRobot, ...agent.trace.map(pointFromTrace)];
  const travelledPoints = tracePoints.slice(0, frame + 2);
  const projectedPoints = tracePoints.slice(frame + 1);
  const deliveredOrderIds = new Set(
    agent.trace
      .slice(0, frame + 1)
      .map((step) => step.deliveredOrderId)
      .filter((orderId): orderId is string => orderId !== null),
  );
  const batteryPercent = Math.max(
    0,
    Math.min(100, (currentStep.battery / data.scenario.batteryCapacity) * 100),
  );
  const traceEndTime = agent.trace[maximumFrame]?.time ?? data.scenario.horizon;
  const completedPercent = maximumFrame ? (frame / maximumFrame) * 100 : 100;
  const activeStyle = { "--agent-color": agent.color } as CSSProperties;
  const stateStatus = currentStep.violations.length
    ? { label: "Constraint violation recorded", tone: "alert" }
    : currentStep.battery <= 0
      ? { label: "Battery depleted", tone: "alert" }
      : batteryPercent <= 20
        ? { label: "Low battery", tone: "warning" }
        : { label: "No active constraint flag", tone: "neutral" };

  const orderState = (order: Order): OrderState => {
    if (deliveredOrderIds.has(order.id)) return "delivered";
    if (currentStep.carriedOrderId === order.id) return "carried";
    if (order.releaseTime > currentTime) return "queued";
    if (currentTime >= data.scenario.horizon) return "expired";
    return "ready";
  };

  const orderStates = data.scenario.orders.map(orderState);
  const deliveredOrderCount = orderStates.filter((state) => state === "delivered").length;
  const expiredOrderCount = orderStates.filter((state) => state === "expired").length;
  const carriedOrderCount = orderStates.filter((state) => state === "carried").length;

  const seekToTime = (time: number) => {
    const nextFrame = agent.trace.findIndex((step) => step.time >= time);
    setPlaying(false);
    setFrame(nextFrame === -1 ? maximumFrame : nextFrame);
  };

  return (
    <main className="site-shell" style={activeStyle}>
      <header className="site-header">
        <a className="wordmark" href="#top">Adaptive Agent Lab</a>
        <nav aria-label="Page navigation">
          <a href="#viewer">Replay</a>
          <a href="#results">Results</a>
          <a href="https://github.com/Revincxt/adaptive-agent-lab">Repository ↗</a>
        </nav>
      </header>

      <article className="research-page">
        <header className="paper-header" id="top">
          <p className="paper-label">Interactive research artifact · {data.scenario.id}</p>
          <h1>Planning and learning in a dynamic warehouse maze</h1>
          <p className="abstract-copy">
            This page replays six planning, reinforcement-learning, and hybrid controllers on one
            shared warehouse episode. Use the viewer to inspect trajectories and state transitions
            under the same order releases and temporary aisle closures.
          </p>
          <div className="artifact-note" role="note">
            <strong>Interpretation.</strong> {data.verificationStatus}. The values below describe a
            committed demonstration tape; they are not a held-out benchmark or an algorithm ranking.
          </div>
          <dl className="study-metadata">
            <div><dt>Grid</dt><dd>{data.scenario.width} × {data.scenario.height}</dd></div>
            <div><dt>Controllers</dt><dd>{data.agents.length}</dd></div>
            <div><dt>Orders</dt><dd>{data.scenario.orders.length}</dd></div>
            <div><dt>Event records</dt><dd>{data.scenario.events.length}</dd></div>
            <div><dt>Horizon</dt><dd>t = {data.scenario.horizon}</dd></div>
          </dl>
        </header>

        <section className="viewer-section" id="viewer" aria-labelledby="viewer-title">
          <div className="section-heading">
            <div>
              <p className="section-number">01</p>
              <h2 id="viewer-title">Paired-episode replay</h2>
              <p>All controllers use the same scenario and recorded event schedule.</p>
            </div>
            <label className="algorithm-select">
              <span>Controller</span>
              <select value={agent.id} onChange={(event) => selectAgent(event.target.value)}>
                {data.agents.map((candidate) => (
                  <option value={candidate.id} key={candidate.id}>{candidate.label}</option>
                ))}
              </select>
            </label>
          </div>

          <div className="operations-grid">
            <figure className="trajectory-figure">
              <header className="figure-heading">
                <div>
                  <span>Figure 1</span>
                  <h3>Scenario trajectory</h3>
                </div>
                <p aria-live={playing ? "off" : "polite"} aria-atomic="true">
                  t = <strong>{currentTime}</strong> / {traceEndTime}
                </p>
              </header>

              <div className="map-wrap">
                <div
                  className="warehouse-map"
                  style={{ aspectRatio: `${data.scenario.width} / ${data.scenario.height}` }}
                  role="img"
                  aria-label={`${data.scenario.width} by ${data.scenario.height} warehouse maze at time ${currentTime}. Robot at column ${robotPosition.x}, row ${robotPosition.y}. ${blocked.size} aisle closures active. ${deliveredOrderCount} orders delivered, ${carriedOrderCount} carried, and ${expiredOrderCount} expired.`}
                >
                  <div
                    className="map-grid"
                    style={{
                      gridTemplateColumns: `repeat(${data.scenario.width}, 1fr)`,
                      gridTemplateRows: `repeat(${data.scenario.height}, 1fr)`,
                    }}
                    aria-hidden="true"
                  >
                    {Array.from({ length: data.scenario.width * data.scenario.height }).map(
                      (_, index) => {
                        const point = {
                          x: index % data.scenario.width,
                          y: Math.floor(index / data.scenario.width),
                        };
                        const key = cellKey(point);
                        const rackIslandLabel = rackIslandLabels.get(key);
                        const pickups = data.scenario.orders.filter((order) => cellKey(order.pickup) === key);
                        const dropoffs = data.scenario.orders.filter((order) => cellKey(order.dropoff) === key);
                        const isRobot = cellKey(robotPosition) === key;
                        const isReceiving = point.x < 2;
                        const isDispatch = point.x >= data.scenario.width - 2;
                        const isPacking = point.y >= data.scenario.height - 2 && !isReceiving && !isDispatch;
                        const zoneClass = chargerSet.has(key)
                          ? "charging-zone"
                          : isReceiving
                            ? "receiving-zone"
                            : isDispatch
                              ? "dispatch-zone"
                              : isPacking
                                ? "packing-zone"
                                : "storage-zone";
                        const zoneTag =
                          point.x === 0 && point.y === 0
                            ? "IN"
                            : point.x === data.scenario.width - 1 && point.y === 0
                              ? "OUT"
                              : point.x === Math.floor(data.scenario.width / 2) && point.y === data.scenario.height - 1
                                ? "PACK"
                                : null;

                        return (
                          <span
                            className={`map-cell ${zoneClass} ${obstacleSet.has(key) ? "obstacle" : ""} ${
                              blocked.has(key) ? "blocked" : ""
                            }`}
                            key={key}
                          >
                            {zoneTag ? <b className="zone-tag">{zoneTag}</b> : null}
                            {rackIslandLabel ? <i className="rack-label">{rackIslandLabel}</i> : null}
                            {chargerSet.has(key) ? <span className="charger">C</span> : null}
                            {pickups.map((order) => (
                              <span
                                className={`order-marker pickup-marker is-${orderState(order) === "carried" ? "picked" : orderState(order)}`}
                                key={`pickup-${order.id}`}
                              >
                                P{data.scenario.orders.indexOf(order) + 1}
                              </span>
                            ))}
                            {dropoffs.map((order) => (
                              <span
                                className={`order-marker dropoff-marker is-${orderState(order)}`}
                                key={`dropoff-${order.id}`}
                              >
                                D{data.scenario.orders.indexOf(order) + 1}
                              </span>
                            ))}
                            {blocked.has(key) ? <span className="closure-marker">×</span> : null}
                            {isRobot ? <span className="robot-marker">R</span> : null}
                          </span>
                        );
                      },
                    )}
                  </div>
                  <RouteLayer points={projectedPoints} width={data.scenario.width} height={data.scenario.height} phase="projected" />
                  <RouteLayer points={travelledPoints} width={data.scenario.width} height={data.scenario.height} phase="travelled" />
                </div>
              </div>

              <div className="map-legend" aria-label="Map legend">
                <span><i className="legend-robot" />robot</span>
                <span><i className="legend-route" />observed path</span>
                <span><i className="legend-projected" />remaining recorded path</span>
                <span><i className="legend-pickup" />pickup</span>
                <span><i className="legend-dropoff" />drop-off</span>
                <span><i className="legend-blocked" />closure</span>
                <span><i className="legend-delivered" />delivered</span>
                <span><i className="legend-expired" />expired</span>
              </div>

              <div className="replay-controls">
                <button
                  className="play-control"
                  onClick={() => {
                    if (frame >= maximumFrame) setFrame(0);
                    setPlaying((value) => !value);
                  }}
                  aria-label={playing ? "Pause trajectory replay" : "Play trajectory replay"}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <div className="timeline-control">
                  <div className="timeline-labels"><span>t = 1</span><span>Episode time</span><span>t = {traceEndTime}</span></div>
                  <div className="range-wrap">
                    <input
                      aria-label={`Replay position, time ${currentTime} of ${traceEndTime}`}
                      aria-valuetext={`t = ${currentTime} of ${traceEndTime}`}
                      type="range"
                      min="0"
                      max={maximumFrame}
                      value={frame}
                      style={{ "--timeline-progress": `${completedPercent}%` } as CSSProperties}
                      onChange={(event) => {
                        setPlaying(false);
                        setFrame(Number(event.target.value));
                      }}
                    />
                    <div className="timeline-events" aria-label="Scenario events">
                      {data.scenario.events
                        .filter((event) => event.time <= traceEndTime)
                        .map((event, index) => (
                          <button
                            key={`${event.kind}-${event.time}-${index}`}
                            className={`timeline-event event-${event.kind}`}
                            style={{
                              left: `${(event.time / traceEndTime) * 100}%`,
                              "--event-lane": index % 2,
                            } as CSSProperties}
                            onClick={() => seekToTime(event.time)}
                            aria-label={`${eventLabels[event.kind]} at time ${event.time}`}
                            title={`${eventLabels[event.kind]} · t=${event.time}`}
                          />
                        ))}
                    </div>
                  </div>
                </div>
                <button className="reset-control" onClick={() => { setPlaying(false); setFrame(0); }}>Reset</button>
              </div>

              <figcaption>
                The solid path has been traversed; the lighter dashed path is the remainder of the
                same committed trajectory. Event markers above the time axis are interactive.
              </figcaption>
            </figure>

            <aside className="state-panel" aria-labelledby="state-title">
              <div className="method-summary">
                <span className="method-swatch" aria-hidden="true" />
                <p>{agent.family}</p>
                <h3>{agent.label}</h3>
                <span>{agent.description}</span>
              </div>
              <h4 id="state-title">State at time t</h4>
              <dl className="state-table">
                <div><dt>Time</dt><dd>{currentTime}</dd></div>
                <div><dt>Position</dt><dd>({robotPosition.x}, {robotPosition.y})</dd></div>
                <div><dt>Applied action</dt><dd>{actionLabels[currentStep.action] ?? currentStep.action}</dd></div>
                <div><dt>Battery</dt><dd>{currentStep.battery} / {data.scenario.batteryCapacity}</dd></div>
                <div><dt>Payload</dt><dd>{currentStep.carriedOrderId ?? "None"}</dd></div>
                <div><dt>Step reward</dt><dd>{currentStep.reward.toFixed(2)}</dd></div>
                <div><dt>Cumulative return</dt><dd>{currentStep.cumulativeReward.toFixed(2)}</dd></div>
                <div><dt>Observed events</dt><dd>{currentStep.eventCount}</dd></div>
              </dl>
              <p className={`state-status is-${stateStatus.tone}`}>{stateStatus.label}</p>
              <div className="method-workload">
                <h4>Method workload</h4>
                <dl>
                  <div><dt>Planning calls</dt><dd>{agent.planningCalls}</dd></div>
                  <div><dt>Expanded nodes</dt><dd>{formatNumber(agent.expandedNodes)}</dd></div>
                  <div><dt>Learning updates</dt><dd>{formatNumber(agent.learningUpdates)}</dd></div>
                </dl>
              </div>
            </aside>
          </div>

          <details className="event-annotations">
            <summary>Environment events ({data.scenario.events.length})</summary>
            <div className="event-table-wrap">
              <table>
                <thead><tr><th>Time</th><th>Event</th><th>Object / coordinate</th><th /></tr></thead>
                <tbody>
                  {data.scenario.events.map((event, index) => {
                    const detail = event.orderId ?? (event.position ? `(${event.position.x}, ${event.position.y})` : "—");
                    return (
                      <tr className={event.time <= currentTime ? "is-observed" : ""} key={`${event.kind}-${event.time}-${index}`}>
                        <td>t = {event.time}</td>
                        <td>{eventLabels[event.kind]}</td>
                        <td>{detail}</td>
                        <td><button onClick={() => seekToTime(event.time)} disabled={event.time > traceEndTime}>View</button></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        </section>

        <section className="results-section" id="results" aria-labelledby="results-title">
          <div className="section-heading">
            <div>
              <p className="section-number">02</p>
              <h2 id="results-title">Paired-episode results</h2>
              <p>Descriptive values from the single committed demonstration artifact.</p>
            </div>
          </div>

          <div className="comparison-shell">
            <table className="comparison-table">
              <caption>
                Non-confirmatory six-controller comparison. Select a controller to replay its trajectory.
              </caption>
              <thead>
                <tr>
                  <th>Controller</th>
                  <th>On time</th>
                  <th>Completion</th>
                  <th>Return</th>
                  <th>Steps</th>
                  <th>Violations</th>
                </tr>
              </thead>
              <tbody>
                {data.agents.map((candidate) => (
                  <tr className={candidate.id === agent.id ? "is-selected" : ""} key={candidate.id}>
                    <th scope="row">
                      <button onClick={() => selectAgent(candidate.id)} aria-pressed={candidate.id === agent.id}>
                        <i style={{ backgroundColor: candidate.color }} />
                        <span><strong>{candidate.label}</strong><small>{candidate.family}</small></span>
                      </button>
                    </th>
                    <td>{formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}</td>
                    <td>{formatPercent(candidate.metrics.weightedCompletionRate)}</td>
                    <td>{formatNumber(candidate.metrics.totalReward)}</td>
                    <td>{candidate.metrics.steps}</td>
                    <td>{candidate.metrics.constraintViolations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            No confidence intervals or held-out estimates are reported here. Confirmatory evaluation
            requires frozen checkpoints, independent test scenarios, and multi-seed analysis.
          </p>
        </section>

        <section className="interpretation-section" aria-labelledby="interpretation-title">
          <div>
            <p className="section-number">03</p>
            <h2 id="interpretation-title">Interpretation notes</h2>
          </div>
          <div className="interpretation-grid">
            <article><h3>Controlled input</h3><p>Every method receives the same scenario, order releases, and closure schedule.</p></article>
            <article><h3>Replay scope</h3><p>The viewer shows committed simulator traces rather than a live scheduler or physical robot.</p></article>
            <article><h3>Claim boundary</h3><p>This artifact supports implementation inspection, not general performance claims.</p></article>
          </div>
        </section>

        <footer className="paper-footer">
          <div>
            <span>Scenario fingerprint</span>
            <code>{data.scenarioFingerprint}</code>
          </div>
          <p>Generated {new Date(data.generatedAt).toLocaleDateString("en-GB")} · source and protocol available on GitHub</p>
        </footer>
      </article>
    </main>
  );
}
