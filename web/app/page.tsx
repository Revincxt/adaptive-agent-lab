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
  pickup: "Secure payload",
  dropoff: "Complete dispatch",
  charge: "Recharge at dock",
  wait: "Hold position",
};

const eventLabels: Record<Event["kind"], string> = {
  order_arrival: "Order released",
  cell_blocked: "Aisle closure",
  cell_unblocked: "Aisle reopened",
};

const cellKey = (point: Point) => `${point.x}:${point.y}`;

const rackIslandLabels = new Map([
  ["2:2", "R01"],
  ["2:7", "R02"],
  ["6:2", "R03"],
  ["6:7", "R04"],
  ["10:2", "R05"],
  ["10:7", "R06"],
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
  const [agentId, setAgentId] = useState("replanning");
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

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
    }, 700 / speed);
    return () => window.clearInterval(timer);
  }, [playing, maximumFrame, speed]);

  const currentStep = agent?.trace[Math.min(frame, maximumFrame)] ?? null;
  const currentTime = currentStep?.time ?? 0;
  const blocked = blockedCells(data?.scenario.events ?? [], currentTime);
  const obstacleSet = new Set((data?.scenario.obstacles ?? []).map(cellKey));
  const chargerSet = new Set((data?.scenario.chargingStations ?? []).map(cellKey));

  if (error) {
    return (
      <main className="loading-shell error-shell">
        <span className="system-label">System / artifact error</span>
        <h1>Mission replay unavailable.</h1>
        <p>{error}</p>
        <a href="https://github.com/Revincxt/adaptive-agent-lab">Inspect the project on GitHub</a>
      </main>
    );
  }

  if (!data || !agent || !currentStep) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-pulse" aria-hidden="true" />
        <span className="system-label">Adaptive Agent Lab</span>
        <h1>Loading reproducible evidence…</h1>
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
  const unitStatus = currentStep.violations.length
    ? { label: "Constraint alert", tone: "alert" }
    : currentStep.battery <= 0
      ? { label: "Battery depleted", tone: "alert" }
      : batteryPercent <= 20
        ? { label: "Energy low", tone: "warning" }
        : { label: "Nominal", tone: "nominal" };

  const orderState = (order: Order): OrderState => {
    if (deliveredOrderIds.has(order.id)) return "delivered";
    if (currentStep.carriedOrderId === order.id) return "carried";
    if (order.releaseTime > currentTime) return "queued";
    if (currentTime >= data.scenario.horizon) return "expired";
    return "ready";
  };

  const seekToTime = (time: number) => {
    const nextFrame = agent.trace.findIndex((step) => step.time >= time);
    setPlaying(false);
    setFrame(nextFrame === -1 ? maximumFrame : nextFrame);
  };

  return (
    <main className="app-shell" style={activeStyle}>
      <header className="command-bar">
        <a className="brand-lockup" href="#mission-control" aria-label="Adaptive Agent Lab home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>
            <strong>Adaptive Agent Lab</strong>
            <small>Warehouse autonomy testbed</small>
          </span>
        </a>
        <div className="command-meta" aria-label="Experiment status">
          <span className="demo-status" title={data.verificationStatus}><i /> {data.verificationStatus}</span>
          <span className="command-divider" aria-hidden="true" />
          <span>Scenario {data.scenario.id}</span>
          <a href="https://github.com/Revincxt/adaptive-agent-lab">GitHub <b aria-hidden="true">↗</b></a>
        </div>
      </header>

      <div className="shell-content">
        <section className="mission-intro" id="mission-control" aria-labelledby="mission-title">
          <div>
            <p className="system-label">Mission control / paired episode 0042</p>
            <h1 id="mission-title">Dynamic warehouse<br /><span>adaptation replay</span></h1>
          </div>
          <div className="intro-brief">
            <p>
              Six controllers enter the same maze of racks, orders, and recorded aisle closures.
              Replay the committed trajectory and inspect how each policy responds.
            </p>
            <dl>
              <div><dt>World</dt><dd>{data.scenario.width} × {data.scenario.height}</dd></div>
              <div><dt>Orders</dt><dd>{data.scenario.orders.length}</dd></div>
              <div><dt>Disruptions</dt><dd>{data.scenario.events.filter((event) => event.kind !== "order_arrival").length}</dd></div>
              <div><dt>Horizon</dt><dd>T+{data.scenario.horizon}</dd></div>
            </dl>
          </div>
        </section>

        <section className="agent-deck" aria-labelledby="agent-deck-title">
          <div className="section-kicker">
            <span>01 / controller matrix</span>
            <h2 id="agent-deck-title">Select an autonomy stack</h2>
            <p>Demo-only weighted completion</p>
          </div>
          <div className="agent-grid">
            {data.agents.map((candidate, index) => (
              <button
                className={`agent-card ${candidate.id === agent.id ? "is-active" : ""}`}
                key={candidate.id}
                onClick={() => selectAgent(candidate.id)}
                aria-pressed={candidate.id === agent.id}
                style={{ "--candidate-color": candidate.color } as CSSProperties}
              >
                <span className="agent-index">{String(index + 1).padStart(2, "0")}</span>
                <span className="agent-name"><strong>{candidate.label}</strong><small>{candidate.family}</small></span>
                <span className="agent-score">{formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}</span>
                <span className="agent-progress" aria-hidden="true"><i style={{ width: formatPercent(candidate.metrics.weightedOnTimeCompletionRate) }} /></span>
              </button>
            ))}
          </div>
        </section>

        <div className="operations-grid">
          <section className="map-console console-panel" aria-labelledby="warehouse-map-title">
            <header className="console-header">
              <div>
                <p className="system-label">02 / recorded floor replay</p>
                <h2 id="warehouse-map-title">Warehouse Maze / Floor 07</h2>
              </div>
              <div className="tick-readout" aria-live={playing ? "off" : "polite"} aria-atomic="true">
                <span>Simulation tick</span>
                <strong>T+{String(currentTime).padStart(3, "0")}</strong>
              </div>
            </header>

            <div className="zone-rail" aria-label="Warehouse zones">
              <span className="zone-receiving"><i />Receiving <small>R01</small></span>
              <span className="zone-storage"><i />Rack maze <small>S02</small></span>
              <span className="zone-packing"><i />Packing <small>P03</small></span>
              <span className="zone-dispatch"><i />Dispatch <small>D04</small></span>
              <span className="zone-charging"><i />Charging <small>C05</small></span>
            </div>

            <div className="map-wrap">
              <div
                className="warehouse-map"
                style={{ aspectRatio: `${data.scenario.width} / ${data.scenario.height}` }}
                role="img"
                aria-label={`${data.scenario.width} by ${data.scenario.height} warehouse maze at tick ${currentTime}. Robot at column ${robotPosition.x}, row ${robotPosition.y}. ${blocked.size} aisle closures active.`}
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
                          ? "RCV"
                          : point.x === data.scenario.width - 1 && point.y === 0
                            ? "DSP"
                            : point.x === Math.floor(data.scenario.width / 2) && point.y === data.scenario.height - 1
                              ? "PCK"
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
                          {chargerSet.has(key) ? <span className="charger"><i />CHG</span> : null}
                          {pickups.map((order) => (
                            <span
                              className={`order-marker pickup-marker is-${orderState(order) === "carried" ? "picked" : orderState(order)}`}
                              key={`pickup-${order.id}`}
                            >
                              <b>P{data.scenario.orders.indexOf(order) + 1}</b><i />
                            </span>
                          ))}
                          {dropoffs.map((order) => (
                            <span className={`order-marker dropoff-marker is-${orderState(order)}`} key={`dropoff-${order.id}`}>
                              <b>D{data.scenario.orders.indexOf(order) + 1}</b><i />
                            </span>
                          ))}
                          {blocked.has(key) ? <span className="closure-gate"><i /><b>BLOCK</b></span> : null}
                          {isRobot ? (
                            <span className={`robot heading-${currentStep.action}`} style={{ backgroundColor: agent.color }}>
                              <i className="robot-heading" />
                              <b>R1</b>
                              <i className="robot-status" />
                            </span>
                          ) : null}
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
              <span><i className="legend-robot" />AMR-01</span>
              <span><i className="legend-route" />travelled route</span>
              <span><i className="legend-projected" />recorded route</span>
              <span><i className="legend-pickup" />pickup</span>
              <span><i className="legend-dropoff" />drop-off</span>
              <span><i className="legend-blocked" />dynamic closure</span>
            </div>

            <div className="timeline-console">
              <div className="transport-controls">
                <button
                  className="play-button"
                  onClick={() => {
                    if (frame >= maximumFrame) setFrame(0);
                    setPlaying((value) => !value);
                  }}
                  aria-label={playing ? "Pause mission replay" : "Play mission replay"}
                >
                  <span aria-hidden="true">{playing ? "Ⅱ" : "▶"}</span>
                </button>
                <div><small>Replay</small><strong>{playing ? "RUNNING" : frame >= maximumFrame ? "COMPLETE" : "PAUSED"}</strong></div>
              </div>
              <div className="timeline-track">
                <div className="timeline-labels"><span>T+001</span><strong>Event tape</strong><span>T+{String(traceEndTime).padStart(3, "0")}</span></div>
                <div className="range-wrap">
                  <input
                    aria-label={`Replay position, tick ${currentTime} of ${traceEndTime}`}
                    aria-valuetext={`T+${currentTime} of T+${traceEndTime}`}
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
                          style={{ left: `${(event.time / traceEndTime) * 100}%` }}
                          onClick={() => seekToTime(event.time)}
                          aria-label={`${eventLabels[event.kind]} at tick ${event.time}`}
                          title={`${eventLabels[event.kind]} · T+${event.time}`}
                        />
                      ))}
                  </div>
                </div>
              </div>
              <button
                className="speed-button"
                onClick={() => setSpeed(speed === 2 ? 0.5 : speed * 2)}
                aria-label={`Replay speed ${speed} times. Change speed.`}
              >
                <small>Speed</small><strong>{speed}×</strong>
              </button>
            </div>
          </section>

          <aside className="telemetry-console console-panel" aria-labelledby="telemetry-title">
            <header className="console-header compact-header">
              <div>
                <p className="system-label">03 / decision telemetry</p>
                <h2 id="telemetry-title">AMR-01 status</h2>
              </div>
              <span className={`unit-status is-${unitStatus.tone}`}><i />{unitStatus.label}</span>
            </header>

            <section className="active-policy">
              <span className="policy-line" aria-hidden="true" />
              <small>Active controller</small>
              <h3>{agent.label}</h3>
              <p>{agent.description}</p>
            </section>

            <section className="decision-readout" aria-live={playing ? "off" : "polite"} aria-atomic="true">
              <div>
                <span>Applied actuator command</span>
                <strong>{actionLabels[currentStep.action] ?? currentStep.action}</strong>
              </div>
              <b className={`action-glyph action-${currentStep.action}`} aria-hidden="true" />
            </section>

            <section className="battery-module" aria-label={`Battery ${Math.round(batteryPercent)} percent`}>
              <div><span>Energy reserve</span><strong>{currentStep.battery}<small> / {data.scenario.batteryCapacity}</small></strong></div>
              <div className="battery-track" aria-hidden="true"><i style={{ width: `${batteryPercent}%` }} /></div>
            </section>

            <dl className="state-grid">
              <div><dt>Grid coordinate</dt><dd>{String(robotPosition.x).padStart(2, "0")} : {String(robotPosition.y).padStart(2, "0")}</dd></div>
              <div><dt>Payload</dt><dd>{currentStep.carriedOrderId ?? "Bay empty"}</dd></div>
              <div><dt>Step reward</dt><dd>{currentStep.reward >= 0 ? "+" : ""}{currentStep.reward.toFixed(2)}</dd></div>
              <div><dt>Total return</dt><dd>{currentStep.cumulativeReward.toFixed(2)}</dd></div>
              <div><dt>Safety flags</dt><dd className={currentStep.violations.length ? "warning-value" : "safe-value"}>{currentStep.violations.length || "CLEAR"}</dd></div>
              <div><dt>Events observed</dt><dd>{currentStep.eventCount}</dd></div>
            </dl>

            <section className="event-log" aria-labelledby="event-log-title">
              <div className="module-heading"><h3 id="event-log-title">Operations event tape</h3><span>{data.scenario.events.length} records</span></div>
              <div className="event-scroll">
                {data.scenario.events.map((event, index) => {
                  const future = event.time > currentTime;
                  const eventPosition = event.position ? ` · ${event.position.x}:${event.position.y}` : "";
                  return (
                    <button
                      className={`event-row event-${event.kind} ${future ? "is-future" : "is-past"}`}
                      key={`${event.kind}-${event.time}-${index}`}
                      onClick={() => seekToTime(event.time)}
                      disabled={event.time > traceEndTime}
                    >
                      <time>T+{String(event.time).padStart(3, "0")}</time>
                      <i aria-hidden="true" />
                      <span><strong>{eventLabels[event.kind]}</strong><small>{event.orderId ?? "floor system"}{eventPosition}</small></span>
                      <b>{future ? "QUEUED" : "LOGGED"}</b>
                    </button>
                  );
                })}
              </div>
            </section>
          </aside>
        </div>

        <section className="evidence-section" aria-labelledby="evidence-title">
          <div className="evidence-heading">
            <div>
              <p className="system-label">04 / paired evaluation</p>
              <h2 id="evidence-title">Adaptation, measured on equal ground.</h2>
            </div>
            <div className="evidence-status"><i />{data.verificationStatus}</div>
          </div>

          <div className="metric-grid">
            <article>
              <span>On-time completion</span>
              <strong>{formatPercent(agent.metrics.weightedOnTimeCompletionRate)}</strong>
              <p>weighted primary metric</p>
              <i style={{ width: formatPercent(agent.metrics.weightedOnTimeCompletionRate) }} />
            </article>
            <article>
              <span>Mission return</span>
              <strong>{formatNumber(agent.metrics.totalReward)}</strong>
              <p>{agent.metrics.completedOrders} / {agent.metrics.totalOrders} orders complete</p>
            </article>
            <article>
              <span>Search workload</span>
              <strong>{formatNumber(agent.expandedNodes)}</strong>
              <p>{agent.planningCalls} planning calls</p>
            </article>
            <article>
              <span>Constraint audit</span>
              <strong className={agent.metrics.constraintViolations ? "warning-value" : "safe-value"}>{agent.metrics.constraintViolations}</strong>
              <p>{agent.metrics.constraintViolations ? "review required" : "no violations recorded"}</p>
            </article>
          </div>

          <div className="comparison-shell">
            <table className="comparison-table">
              <caption>Demo-only six-agent paired replay comparison. Select an agent name to inspect its trajectory.</caption>
              <thead>
                <tr><th>Controller</th><th>On time</th><th>Completion</th><th>Return</th><th>Steps</th><th>Violations</th></tr>
              </thead>
              <tbody>
                {data.agents.map((candidate) => (
                  <tr className={candidate.id === agent.id ? "is-selected" : ""} key={candidate.id}>
                    <th scope="row">
                      <button
                        onClick={() => selectAgent(candidate.id)}
                        aria-pressed={candidate.id === agent.id}
                      >
                        <i style={{ backgroundColor: candidate.color }} />
                        <span><strong>{candidate.label}</strong><small>{candidate.family}</small></span>
                      </button>
                    </th>
                    <td><strong>{formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}</strong><span className="table-meter" aria-hidden="true"><i style={{ width: formatPercent(candidate.metrics.weightedOnTimeCompletionRate), backgroundColor: candidate.color }} /></span></td>
                    <td>{formatPercent(candidate.metrics.weightedCompletionRate)}</td>
                    <td>{formatNumber(candidate.metrics.totalReward)}</td>
                    <td>{candidate.metrics.steps}</td>
                    <td className={candidate.metrics.constraintViolations ? "warning-value" : "safe-value"}>{candidate.metrics.constraintViolations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <footer className="site-footer">
        <div>
          <span className="system-label">Reproducibility fingerprint</span>
          <code>{data.scenarioFingerprint}</code>
        </div>
        <p>Committed evidence · paired tape · no live scheduler<br />Generated {new Date(data.generatedAt).toLocaleDateString("en-GB")}</p>
      </footer>
    </main>
  );
}
