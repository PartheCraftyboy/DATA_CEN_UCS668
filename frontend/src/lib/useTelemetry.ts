import { useCallback, useEffect, useRef, useState } from "react";
import { OfflineSimulator, offlineHandshake } from "./offlineSim";
import type {
  Command,
  ConnectionState,
  ConveyorItem,
  Handshake,
  Telemetry,
} from "./types";

const WS_URL =
  (import.meta.env?.PUBLIC_WS_URL as string | undefined) ?? "ws://localhost:8000/ws";

/** How long to wait for the backend before falling back to the local mirror. */
const CONNECT_TIMEOUT_MS = 2500;
/** Reconnect backoff ceiling. */
const RECONNECT_MAX_MS = 8000;

export interface TelemetryAPI {
  connection: ConnectionState;
  handshake: Handshake | null;
  telemetry: Telemetry | null;
  /** The conveyor belt, assembled client-side from the server's deltas. */
  conveyor: ConveyorItem[];
  send: (command: Command) => void;
  reconnect: () => void;
}

/**
 * Subscribes to the backend telemetry stream, with a local fallback.
 *
 * Connection lifecycle:
 *
 *   connecting → live        the FastAPI WebSocket answered
 *   connecting → simulated   it did not, so the in-browser mirror took over
 *   live       → offline     the socket dropped mid-run; retry with backoff
 *
 * The cockpit always has data to render. The connection badge always says
 * which source produced it, so a simulated run is never passed off as a
 * measured one.
 *
 * The conveyor is assembled here rather than on the server: the stream carries
 * only newly completed frames, and this hook maintains the ring buffer. That
 * keeps the socket at ~0.3 MB/s instead of ~6 MB/s, which is the difference
 * between a smooth 25 FPS belt and a stuttering one.
 */
export function useTelemetry(): TelemetryAPI {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [handshake, setHandshake] = useState<Handshake | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [conveyor, setConveyor] = useState<ConveyorItem[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const simRef = useRef<OfflineSimulator | null>(null);
  const simTimerRef = useRef<number | null>(null);
  const retryRef = useRef(0);
  const disposedRef = useRef(false);

  // ── conveyor assembly ──────────────────────────────────────────────────────

  const absorb = useCallback((payload: Telemetry) => {
    setTelemetry(payload);
    if (payload.conveyor_new?.length) {
      setConveyor((prev) => {
        const next = prev.concat(payload.conveyor_new);
        const depth = payload.conveyor_depth ?? 14;
        return next.length > depth ? next.slice(next.length - depth) : next;
      });
    }
  }, []);

  // ── offline mirror ─────────────────────────────────────────────────────────

  const startSimulation = useCallback(() => {
    if (simTimerRef.current !== null) return;
    const sim = new OfflineSimulator();
    simRef.current = sim;
    setHandshake(offlineHandshake());
    setConnection("simulated");

    // setInterval rather than rAF: the simulation must advance at a fixed
    // 25 Hz regardless of display refresh rate or whether the tab is visible.
    // Tying it to rAF would make the queue dynamics depend on the monitor,
    // which would be an absurd property for a latency simulator to have.
    simTimerRef.current = window.setInterval(() => {
      absorb(sim.step());
    }, 1000 / 25);
  }, [absorb]);

  const stopSimulation = useCallback(() => {
    if (simTimerRef.current !== null) {
      window.clearInterval(simTimerRef.current);
      simTimerRef.current = null;
    }
    simRef.current = null;
  }, []);

  // ── socket ─────────────────────────────────────────────────────────────────

  const connect = useCallback(() => {
    if (disposedRef.current) return;

    let settled = false;
    let socket: WebSocket;
    try {
      socket = new WebSocket(WS_URL);
    } catch {
      startSimulation();
      return;
    }
    socketRef.current = socket;

    const timeout = window.setTimeout(() => {
      if (!settled) {
        settled = true;
        try {
          socket.close();
        } catch {
          /* already closing */
        }
        startSimulation();
      }
    }, CONNECT_TIMEOUT_MS);

    socket.onopen = () => {
      settled = true;
      window.clearTimeout(timeout);
      retryRef.current = 0;
      stopSimulation();
      setConnection("live");
    };

    socket.onmessage = (event) => {
      let msg: Telemetry | Handshake | { type: string; [k: string]: unknown };
      try {
        msg = JSON.parse(event.data as string);
      } catch {
        return;
      }

      switch (msg.type) {
        case "hello":
          setHandshake(msg as Handshake);
          setConveyor([]);
          break;
        case "telemetry":
          absorb(msg as Telemetry);
          break;
        case "calibration":
          // A re-measured inference latency, arriving out of band. Folded into
          // the current telemetry so the readout updates without waiting for
          // the next tick.
          setTelemetry((prev) =>
            prev
              ? {
                  ...prev,
                  model: {
                    ...prev.model,
                    measured_latency_ms: Number(
                      (msg as { measured_latency_ms: number }).measured_latency_ms,
                    ),
                  },
                }
              : prev,
          );
          break;
        default:
          break;
      }
    };

    socket.onerror = () => {
      /* onclose always follows; handled there. */
    };

    socket.onclose = () => {
      window.clearTimeout(timeout);
      socketRef.current = null;
      if (disposedRef.current) return;

      if (!settled) {
        // Never connected at all — the backend is not running.
        settled = true;
        startSimulation();
        return;
      }

      // Was live and dropped. Retry with exponential backoff, and run the
      // mirror in the meantime so the screen does not freeze mid-demo.
      setConnection("offline");
      startSimulation();
      retryRef.current += 1;
      const delay = Math.min(RECONNECT_MAX_MS, 600 * 2 ** (retryRef.current - 1));
      window.setTimeout(connect, delay);
    };
  }, [absorb, startSimulation, stopSimulation]);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
      stopSimulation();
      const s = socketRef.current;
      socketRef.current = null;
      if (s && (s.readyState === WebSocket.OPEN || s.readyState === WebSocket.CONNECTING)) {
        s.close();
      }
    };
    // Mount only: connect() is stable via useCallback and re-running would
    // churn the socket.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── command channel ────────────────────────────────────────────────────────

  const send = useCallback((command: Command) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(command));
      return;
    }

    // Apply locally against the mirror.
    const sim = simRef.current;
    if (!sim) return;
    switch (command.action) {
      case "set_model":
        sim.modelKey = command.model;
        break;
      case "set_storm":
        sim.storms[command.storm] = Math.min(1, Math.max(0, command.intensity));
        break;
      case "toggle_storm":
        sim.storms[command.storm] = sim.storms[command.storm] > 0 ? 0 : 0.75;
        break;
      case "set_filter":
        if (command.enabled) sim.filters.add(command.filter);
        else sim.filters.delete(command.filter);
        break;
      case "toggle_filter":
        if (sim.filters.has(command.filter)) sim.filters.delete(command.filter);
        else sim.filters.add(command.filter);
        break;
      case "pause":
        sim.paused = command.paused;
        break;
      case "reset":
        sim.reset();
        setConveyor([]);
        break;
      default:
        break;
    }
  }, []);

  const reconnect = useCallback(() => {
    retryRef.current = 0;
    setConnection("connecting");
    connect();
  }, [connect]);

  return { connection, handshake, telemetry, conveyor, send, reconnect };
}
