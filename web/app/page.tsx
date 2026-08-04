"use client";

import { useEffect, useState } from "react";

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

const actionLabels: Record<string, string> = {
  up: "move north",
  down: "move south",
  left: "move west",
  right: "move east",
  pickup: "pick up order",
  dropoff: "deliver order",
  charge: "recharge",
  wait: "hold position",
};

const cellKey = (point: Point) => `${point.x}:${point.y}`;

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

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 1 }).format(value);
}

export default function Home() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentId, setAgentId] = useState("replanning");
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    fetch("/demo-data.json")
      .then((response) => {
        if (!response.ok) throw new Error(`demo artifact returned ${response.status}`);
        return response.json() as Promise<DemoData>;
      })
      .then((payload) => {
        setData(payload);
        const preferred = payload.agents.find((agent) => agent.id === "hybrid");
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
      <main className="loading-shell">
        <p className="eyebrow">Adaptive Agent Lab</p>
        <h1>The experiment artifact is not ready yet.</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!data || !agent || !currentStep) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-pulse" />
        <p className="eyebrow">Adaptive Agent Lab</p>
        <h1>Loading reproducible evidence…</h1>
      </main>
    );
  }

  const robotPosition = { x: currentStep.position[0], y: currentStep.position[1] };
  const recentEvents = data.scenario.events
    .filter((event) => event.time <= currentTime)
    .slice(-4)
    .reverse();

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">A</span>
          <div>
            <p className="brand-name">Adaptive Agent Lab</p>
            <p className="brand-subtitle">Planning × learning under disruption</p>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="status-dot" />
          <span>{data.verificationStatus}</span>
          <span className="meta-rule" />
          <span>{data.scenario.id}</span>
        </div>
      </header>

      <section className="hero-strip">
        <div>
          <p className="eyebrow">Paired episode replay · seed 42</p>
          <h1>When the aisle closes,<br />what adapts first?</h1>
        </div>
        <p className="hero-copy">
          Six policies face the same orders, closure tape, transition rules, and
          constraint audit. Select a method, scrub the timeline, and inspect every decision.
        </p>
      </section>

      <div className="workspace-grid">
        <aside className="agent-panel panel">
          <div className="panel-heading">
            <span>01</span>
            <div>
              <p className="eyebrow">Method roster</p>
              <h2>Agent families</h2>
            </div>
          </div>
          <div className="agent-list" role="listbox" aria-label="Agent result">
            {data.agents.map((candidate) => (
              <button
                className={`agent-row ${candidate.id === agent.id ? "is-active" : ""}`}
                key={candidate.id}
                onClick={() => selectAgent(candidate.id)}
                role="option"
                aria-selected={candidate.id === agent.id}
              >
                <span className="agent-swatch" style={{ background: candidate.color }} />
                <span className="agent-identity">
                  <strong>{candidate.label}</strong>
                  <small>{candidate.family}</small>
                </span>
                <span className="agent-score">
                  {formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}
                </span>
              </button>
            ))}
          </div>
          <div className="method-note">
            <span className="agent-swatch large" style={{ background: agent.color }} />
            <p>{agent.description}</p>
          </div>
        </aside>

        <section className="mission-panel panel">
          <div className="panel-heading mission-heading">
            <span>02</span>
            <div>
              <p className="eyebrow">Mission replay</p>
              <h2>{agent.label}</h2>
            </div>
            <div className="tick-readout">
              <small>tick</small>
              <strong>{String(currentTime).padStart(3, "0")}</strong>
              <span>/ {data.scenario.horizon}</span>
            </div>
          </div>

          <div className="map-wrap">
            <div
              className="warehouse-map"
              style={{
                gridTemplateColumns: `repeat(${data.scenario.width}, 1fr)`,
                gridTemplateRows: `repeat(${data.scenario.height}, 1fr)`,
              }}
              aria-label={`Warehouse at tick ${currentTime}`}
            >
              {Array.from({ length: data.scenario.width * data.scenario.height }).map(
                (_, index) => {
                  const point = {
                    x: index % data.scenario.width,
                    y: Math.floor(index / data.scenario.width),
                  };
                  const key = cellKey(point);
                  const pickup = data.scenario.orders.find(
                    (order) => cellKey(order.pickup) === key,
                  );
                  const dropoff = data.scenario.orders.find(
                    (order) => cellKey(order.dropoff) === key,
                  );
                  const isRobot = cellKey(robotPosition) === key;
                  return (
                    <div
                      className={`map-cell ${obstacleSet.has(key) ? "obstacle" : ""} ${
                        blocked.has(key) ? "blocked" : ""
                      }`}
                      key={key}
                    >
                      {chargerSet.has(key) && <span className="charger">C</span>}
                      {pickup && <span className="pickup">P{pickup.id.slice(-1)}</span>}
                      {dropoff && <span className="dropoff">D{dropoff.id.slice(-1)}</span>}
                      {isRobot && (
                        <span className="robot" style={{ background: agent.color }}>
                          <i />
                        </span>
                      )}
                    </div>
                  );
                },
              )}
            </div>
            <div className="map-legend">
              <span><i className="legend-robot" /> robot</span>
              <span><i className="legend-pickup" /> pickup</span>
              <span><i className="legend-dropoff" /> drop-off</span>
              <span><i className="legend-blocked" /> closure</span>
            </div>
          </div>

          <div className="timeline-controls">
            <button
              className="play-button"
              onClick={() => {
                if (frame >= maximumFrame) setFrame(0);
                setPlaying((value) => !value);
              }}
              aria-label={playing ? "Pause replay" : "Play replay"}
            >
              {playing ? "Ⅱ" : "▶"}
            </button>
            <input
              aria-label="Replay position"
              type="range"
              min="0"
              max={maximumFrame}
              value={frame}
              onChange={(event) => {
                setPlaying(false);
                setFrame(Number(event.target.value));
              }}
            />
            <button className="speed-button" onClick={() => setSpeed(speed === 2 ? 0.5 : speed * 2)}>
              {speed}×
            </button>
          </div>
        </section>

        <aside className="inspection-panel panel">
          <div className="panel-heading">
            <span>03</span>
            <div>
              <p className="eyebrow">Decision inspection</p>
              <h2>State & events</h2>
            </div>
          </div>
          <div className="decision-card">
            <small>Selected action</small>
            <strong>{actionLabels[currentStep.action] ?? currentStep.action}</strong>
            <p>
              reward <b>{currentStep.reward.toFixed(2)}</b>
              <span /> cumulative <b>{currentStep.cumulativeReward.toFixed(2)}</b>
            </p>
          </div>
          <dl className="state-grid">
            <div><dt>Battery</dt><dd>{currentStep.battery} / {data.scenario.batteryCapacity}</dd></div>
            <div><dt>Payload</dt><dd>{currentStep.carriedOrderId ?? "empty"}</dd></div>
            <div><dt>Position</dt><dd>{robotPosition.x}, {robotPosition.y}</dd></div>
            <div><dt>Violations</dt><dd>{currentStep.violations.length}</dd></div>
          </dl>
          <div className="event-log">
            <div className="subheading"><span>Event tape</span><small>latest first</small></div>
            {recentEvents.length ? recentEvents.map((event, index) => (
              <div className="event-row" key={`${event.time}-${event.kind}-${index}`}>
                <time>T+{event.time}</time>
                <span className={`event-icon ${event.kind}`} />
                <p>{event.kind.replaceAll("_", " ")}</p>
              </div>
            )) : <p className="empty-log">No external event has fired.</p>}
          </div>
        </aside>
      </div>

      <section className="evidence-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Release evidence</p>
            <h2>Same world. Different adaptation cost.</h2>
          </div>
          <p>Primary metric: weighted on-time completion rate. Timing is descriptive, not a reward term.</p>
        </div>
        <div className="metric-cards">
          <article>
            <small>On-time completion</small>
            <strong>{formatPercent(agent.metrics.weightedOnTimeCompletionRate)}</strong>
            <div className="metric-bar"><i style={{ width: formatPercent(agent.metrics.weightedOnTimeCompletionRate), background: agent.color }} /></div>
          </article>
          <article>
            <small>Total return</small>
            <strong>{formatNumber(agent.metrics.totalReward)}</strong>
            <p>{agent.metrics.completedOrders} of {agent.metrics.totalOrders} orders delivered</p>
          </article>
          <article>
            <small>Adaptation work</small>
            <strong>{formatNumber(agent.expandedNodes)}</strong>
            <p>{agent.planningCalls} plans · {agent.learningUpdates} learning updates</p>
          </article>
          <article>
            <small>Constraint audit</small>
            <strong>{agent.metrics.constraintViolations}</strong>
            <p>{agent.metrics.constraintViolations === 0 ? "no violations recorded" : "review required"}</p>
          </article>
        </div>

        <div className="comparison-table">
          <div className="table-row table-head">
            <span>method</span><span>on time</span><span>return</span><span>steps</span><span>violations</span>
          </div>
          {data.agents.map((candidate) => (
            <button
              className="table-row"
              key={candidate.id}
              onClick={() => selectAgent(candidate.id)}
            >
              <span><i style={{ background: candidate.color }} />{candidate.label}</span>
              <strong>{formatPercent(candidate.metrics.weightedOnTimeCompletionRate)}</strong>
              <span>{formatNumber(candidate.metrics.totalReward)}</span>
              <span>{candidate.metrics.steps}</span>
              <span>{candidate.metrics.constraintViolations}</span>
            </button>
          ))}
        </div>
      </section>

      <footer>
        <div>
          <p className="eyebrow">Reproducibility fingerprint</p>
          <code>{data.scenarioFingerprint}</code>
        </div>
        <p>Generated {new Date(data.generatedAt).toLocaleDateString("en-GB")} · no live scheduler · committed evidence only</p>
      </footer>
    </main>
  );
}
