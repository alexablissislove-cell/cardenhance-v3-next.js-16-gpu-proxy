"use client";

import {
  AlertCircle,
  ArrowDownToLine,
  ArrowUpRight,
  Bot,
  Check,
  Cpu,
  Download,
  FileJson,
  FileSpreadsheet,
  LayoutDashboard,
  Library,
  LoaderCircle,
  Menu,
  Play,
  RefreshCw,
  RotateCcw,
  ScanLine,
  Settings as SettingsIcon,
  Sliders,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ComparisonView } from "@/components/comparison-view";
import {
  runAdaptiveCardPipeline,
  splitMultiCardSheet,
  type AdaptiveAuditMetrics,
  type RotationQuadrant,
} from "@/lib/adaptiveCardPipeline";
import {
  DEFAULT_MAX_IMAGE_BYTES,
  IMAGE_TYPES,
  ImageDimensions,
  OutputFormat,
  ServiceStatus,
} from "@/lib/types";

const initialStatus: ServiceStatus = {
  configured: false,
  ready: false,
  maxBytes: DEFAULT_MAX_IMAGE_BYTES,
  detail: "Checking inference service",
};

interface BatchCardItem {
  id: string;
  filename: string;
  file: File;
  rawUrl: string;
  processedUrl: string;
  selected: boolean;
  enhanced: boolean;
  gpuEnhanced: boolean;
  manualRotationDeg: RotationQuadrant | null;
  rawPixels: Uint8ClampedArray;
  rawW: number;
  rawH: number;
  audit: AdaptiveAuditMetrics;
  inferenceMs?: number | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${(bytes / 1_000_000).toFixed(2)} MB`;
}

async function requestServiceStatus(): Promise<ServiceStatus> {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    return (await response.json()) as ServiceStatus;
  } catch {
    return { ...initialStatus, detail: "Local Adaptive Mode active (GPU endpoint not connected)" };
  }
}

export function RestorationStudio() {
  const inputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);

  // Navigation across the 7 approved routes
  const [activePage, setActivePage] = useState<
    "dashboard" | "studio" | "library" | "detail" | "audit" | "assistant" | "settings"
  >("studio");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Studio Sliders & Mode State (Exact Obsidian spec defaults)
  const [adaptiveMode, setAdaptiveMode] = useState(true);
  const [scale, setScale] = useState(2.0);
  const [descratch, setDescratch] = useState(0.35);
  const [onlyChromeDescratch, setOnlyChromeDescratch] = useState(false);
  const [denoise, setDenoise] = useState(0.20);
  const [sharpen, setSharpen] = useState(0.50);
  const [contrast, setContrast] = useState(0.08);
  const [splitSheets, setSplitSheets] = useState(true);

  // GPU FastAPI Tiled Inference Settings (from d78fb64)
  const [service, setService] = useState<ServiceStatus>(initialStatus);
  const [checking, setChecking] = useState(true);
  const [tileSize, setTileSize] = useState(512);
  const [tileOverlap, setTileOverlap] = useState(64);
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("png");
  const [amp, setAmp] = useState(true);

  // Batch & Single Card State
  const [cards, setCards] = useState<BatchCardItem[]>([]);
  const [activeCardId, setActiveCardId] = useState<string>("");
  const [processing, setProcessing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [toastMsg, setToastMsg] = useState("");

  // 3D Holographic Viewer State
  const [tilt, setTilt] = useState({ rx: -8, ry: 14, glareX: 38, glareY: 32 });
  const [compareBefore, setCompareBefore] = useState(false);
  const [holoFoilEnabled, setHoloFoilEnabled] = useState(true);
  const [dimensions, setDimensions] = useState<ImageDimensions | null>(null);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 3200);
  }, []);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    setService(await requestServiceStatus());
    setChecking(false);
  }, []);

  useEffect(() => {
    let active = true;
    void requestServiceStatus().then((status) => {
      if (!active) return;
      setService(status);
      setChecking(false);
    });
    return () => {
      active = false;
    };
  }, []);

  // Process a raw RGBA buffer through the local TypeScript adaptive pipeline
  const processBufferLocal = useCallback(
    async (
      rgba: Uint8ClampedArray,
      width: number,
      height: number,
      manualRotationDeg: RotationQuadrant | null = null
    ) => {
      const result = await runAdaptiveCardPipeline(rgba, width, height, {
        mode: adaptiveMode ? "adaptive" : "manual",
        scale,
        maxLongEdge: 2200,
        descratch,
        onlyChromeDescratch,
        denoise,
        sharpen,
        contrast,
        manualRotationDeg,
      });
      const outCanvas = document.createElement("canvas");
      outCanvas.width = result.width;
      outCanvas.height = result.height;
      const ctx = outCanvas.getContext("2d");
      if (ctx) {
        const copy = new Uint8ClampedArray(result.rgba);
        ctx.putImageData(new ImageData(copy, result.width, result.height), 0, 0);
      }
      return {
        processedUrl: outCanvas.toDataURL("image/png"),
        audit: result.audit,
      };
    },
    [adaptiveMode, scale, descratch, onlyChromeDescratch, denoise, sharpen, contrast]
  );

  // Ingest one or more uploaded files (supports 2x2 / 3x3 sheet splitting)
  const ingestFiles = useCallback(
    async (fileList: File[]) => {
      setError("");
      if (!fileList.length) return;
      setProcessing(true);
      const newItems: BatchCardItem[] = [];

      for (const candidate of fileList) {
        if (!IMAGE_TYPES.includes(candidate.type as (typeof IMAGE_TYPES)[number])) {
          setError("Choose a JPEG, PNG, or WebP image.");
          continue;
        }
        if (candidate.size > service.maxBytes) {
          setError(
            `Image ${candidate.name} exceeds the ${formatBytes(service.maxBytes)} Vercel upload limit.`
          );
          continue;
        }

        const rawUrl = URL.createObjectURL(candidate);
        try {
          const img = new window.Image();
          await new Promise<void>((res, rej) => {
            img.onload = () => res();
            img.onerror = () => rej(new Error("Decode error"));
            img.src = rawUrl;
          });

          setDimensions({ width: img.naturalWidth, height: img.naturalHeight });

          const maxDecodeEdge = 1400;
          const s = Math.min(1, maxDecodeEdge / Math.max(img.naturalWidth, img.naturalHeight));
          const dw = Math.max(16, Math.round(img.naturalWidth * s));
          const dh = Math.max(16, Math.round(img.naturalHeight * s));
          const canvas = document.createElement("canvas");
          canvas.width = dw;
          canvas.height = dh;
          const ctx = canvas.getContext("2d");
          if (!ctx) continue;
          ctx.drawImage(img, 0, 0, dw, dh);
          const imgData = ctx.getImageData(0, 0, dw, dh);

          const slices = splitSheets
            ? splitMultiCardSheet(imgData.data, dw, dh)
            : [
                {
                  rgba: imgData.data,
                  width: dw,
                  height: dh,
                  index: 0,
                  bounds: { x: 0, y: 0, width: dw, height: dh },
                },
              ];

          for (const slice of slices) {
            const { processedUrl, audit } = await processBufferLocal(
              slice.rgba,
              slice.width,
              slice.height,
              null
            );
            const sliceName =
              slices.length > 1
                ? `${candidate.name.replace(/\.[^.]+$/, "")}-card-${slice.index + 1}.png`
                : candidate.name;
            newItems.push({
              id: `card-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              filename: sliceName,
              file: candidate,
              rawUrl,
              processedUrl,
              selected: true,
              enhanced: true,
              gpuEnhanced: false,
              manualRotationDeg: null,
              rawPixels: new Uint8ClampedArray(slice.rgba),
              rawW: slice.width,
              rawH: slice.height,
              audit,
            });
          }
        } catch {
          setError(`Could not decode ${candidate.name} as an image.`);
        }
      }

      if (newItems.length > 0) {
        setCards((prev) => {
          const combined = [...newItems, ...prev];
          return combined;
        });
        setActiveCardId((prev) => prev || newItems[0].id);
        showToast(`Restored ${newItems.length} card scan(s) upright in Adaptive Mode`);
      }
      setProcessing(false);
    },
    [service.maxBytes, splitSheets, processBufferLocal, showToast]
  );

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    void ingestFiles(files);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files || []);
    void ingestFiles(files);
  }

  // Re-run local adaptive restoration on target card IDs
  const runBatchLocal = useCallback(
    async (
      targetIds: string[],
      rotationOverrides: Record<string, RotationQuadrant | null> = {}
    ) => {
      if (!targetIds.length) return;
      setProcessing(true);
      const updated: BatchCardItem[] = [];
      for (const card of cards) {
        if (!targetIds.includes(card.id)) {
          updated.push(card);
          continue;
        }
        const nextRot =
          rotationOverrides[card.id] !== undefined
            ? rotationOverrides[card.id]
            : card.manualRotationDeg;
        const { processedUrl, audit } = await processBufferLocal(
          card.rawPixels,
          card.rawW,
          card.rawH,
          nextRot
        );
        updated.push({
          ...card,
          enhanced: true,
          manualRotationDeg: nextRot,
          processedUrl,
          audit,
        });
      }
      setCards(updated);
      setProcessing(false);
    },
    [cards, processBufferLocal]
  );

  // Optional Server GPU /api/restore pass (when FastAPI PyTorch backend is connected)
  const runGpuRestoreOnSelected = useCallback(async () => {
    const selected = cards.filter((c) => c.selected);
    if (!selected.length || processing) return;

    if (!service.configured || !service.ready) {
      // Run local browser adaptive pipeline when GPU backend is not configured
      await runBatchLocal(selected.map((c) => c.id));
      showToast(`Adaptive browser restoration completed on ${selected.length} selected card(s)`);
      return;
    }

    setError("");
    setProcessing(true);
    const controller = new AbortController();
    requestRef.current = controller;

    try {
      for (const target of selected) {
        const respBlob = await fetch(target.processedUrl).then((r) => r.blob());
        const body = new FormData();
        body.append("image", new File([respBlob], target.filename, { type: "image/png" }));
        body.append("tileSize", String(tileSize));
        body.append("tileOverlap", String(tileOverlap));
        body.append("outputFormat", outputFormat);
        body.append("amp", String(amp));

        const response = await fetch("/api/restore", {
          method: "POST",
          body,
          signal: controller.signal,
        });
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error ?? `GPU restoration failed with HTTP ${response.status}`);
        }
        const restored = await response.blob();
        if (!restored.type.startsWith("image/")) {
          throw new Error("The GPU service returned an invalid image.");
        }
        const gpuUrl = URL.createObjectURL(restored);
        const elapsed = Number(response.headers.get("x-inference-ms"));
        setCards((prev) =>
          prev.map((c) =>
            c.id === target.id
              ? {
                  ...c,
                  processedUrl: gpuUrl,
                  gpuEnhanced: true,
                  inferenceMs: Number.isFinite(elapsed) && elapsed > 0 ? elapsed : null,
                }
              : c
          )
        );
      }
      showToast(`Completed RaGAN GPU restoration on ${selected.length} card(s)`);
    } catch (reason) {
      if (reason instanceof Error && reason.name === "AbortError") {
        setError("Restoration cancelled. Your source scan is unchanged.");
      } else {
        setError(reason instanceof Error ? reason.message : "Restoration failed unexpectedly.");
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setProcessing(false);
      }
    }
  }, [
    cards,
    processing,
    service.configured,
    service.ready,
    tileSize,
    tileOverlap,
    outputFormat,
    amp,
    runBatchLocal,
    showToast,
  ]);

  const handleQuickRotate = (cardId: string, deltaDeg: number) => {
    const target = cards.find((c) => c.id === cardId);
    if (!target) return;
    const curDeg = target.manualRotationDeg ?? target.audit.appliedRotationDeg ?? 0;
    const nextDeg = ((((curDeg + deltaDeg) % 360) + 360) % 360) as RotationQuadrant;
    void runBatchLocal([cardId], { [cardId]: nextDeg });
    showToast(`Rotated ${target.filename} to ${nextDeg}° CW`);
  };

  const handleBatchRotate = (deltaDeg: number) => {
    const selected = cards.filter((c) => c.selected);
    if (!selected.length) return;
    const map: Record<string, RotationQuadrant> = {};
    for (const c of selected) {
      const curDeg = c.manualRotationDeg ?? c.audit.appliedRotationDeg ?? 0;
      map[c.id] = ((((curDeg + deltaDeg) % 360) + 360) % 360) as RotationQuadrant;
    }
    void runBatchLocal(
      selected.map((c) => c.id),
      map
    );
    showToast(`Rotated ${selected.length} selected card(s) by ${deltaDeg > 0 ? "+" : ""}${deltaDeg}°`);
  };

  const handleDownloadEnhancedPngs = () => {
    const selected = cards.filter((c) => c.selected);
    if (!selected.length) {
      showToast("Select at least 1 card to download enhanced PNGs");
      return;
    }
    selected.forEach((card, idx) => {
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = card.processedUrl;
        a.download = `${card.filename.replace(/\.[^.]+$/, "")}-enhanced.${
          outputFormat === "png" ? "png" : "jpg"
        }`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, idx * 150);
    });
    showToast(`Downloading ${selected.length} enhanced card image(s)`);
  };

  const handleExportAuditJson = () => {
    const payload = {
      exportedAt: new Date().toISOString(),
      engine: "CardEnhance v3.0 Obsidian Adaptive + RaGAN GPU Studio",
      mode: adaptiveMode ? "adaptive" : "manual",
      gpuServiceConnected: service.configured && service.ready,
      totalScans: cards.length,
      selectedScans: cards.filter((c) => c.selected).length,
      cards: cards.map((c) => ({
        filename: c.filename,
        audit: c.audit,
        gpuEnhanced: c.gpuEnhanced,
        inferenceMs: c.inferenceMs ?? null,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cardenhance-audit-telemetry.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Exported real audit telemetry JSON");
  };

  const counts = useMemo(() => {
    const total = cards.length;
    const selected = cards.filter((c) => c.selected).length;
    const enhanced = cards.filter((c) => c.enhanced).length;
    const chromeCount = cards.filter((c) => c.audit.isChromeFoil).length;
    const avgDelta = total
      ? (cards.reduce((s, c) => s + c.audit.sharpnessDeltaPct, 0) / total).toFixed(1)
      : "0.0";
    return { total, selected, enhanced, chromeCount, avgDelta };
  }, [cards]);

  const activeCard = useMemo(
    () => cards.find((c) => c.id === activeCardId) || cards[0] || null,
    [cards, activeCardId]
  );

  const ready = service.configured && service.ready;

  const navItems = [
    { id: "dashboard", label: "Dashboard", tag: "01", icon: LayoutDashboard },
    { id: "studio", label: "Studio", tag: "02", icon: Sliders },
    { id: "library", label: "Library", tag: "03", icon: Library },
    { id: "detail", label: "Card Detail (3D)", tag: "04", icon: Sparkles },
    { id: "audit", label: "Audit", tag: "05", icon: FileSpreadsheet },
    { id: "assistant", label: "Assistant", tag: "06", icon: Bot },
    { id: "settings", label: "Settings", tag: "07", icon: SettingsIcon },
  ] as const;

  return (
    <div
      className="min-h-screen flex flex-col lg:flex-row font-sans selection:bg-[#00FFFF]/20 selection:text-[#00FFFF]"
      style={{
        backgroundColor: "#080808",
        color: "#EAFFFB",
        backgroundImage:
          "linear-gradient(to right, rgba(10, 61, 61, 0.16) 1px, transparent 1px), linear-gradient(to bottom, rgba(10, 61, 61, 0.16) 1px, transparent 1px)",
        backgroundSize: "32px 32px",
      }}
    >
      {/* Toast Notification */}
      {toastMsg && (
        <div
          className="fixed bottom-5 right-5 z-50 flex items-center gap-2.5 px-4 py-2.5 font-mono text-xs"
          style={{
            backgroundColor: "#0C1210",
            border: "1px solid #00FFFF",
            color: "#EAFFFB",
            boxShadow: "0 0 20px rgba(0, 255, 255, 0.25)",
          }}
        >
          <span className="h-2 w-2 rounded-full bg-[#00FF66]" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* Hidden File Inputs */}
      <input
        ref={inputRef}
        className="hidden"
        type="file"
        accept={IMAGE_TYPES.join(",")}
        multiple
        onChange={onInputChange}
      />
      <input
        ref={batchInputRef}
        className="hidden"
        type="file"
        accept={IMAGE_TYPES.join(",")}
        multiple
        onChange={onInputChange}
      />

      {/* Mobile Top Bar */}
      <div
        className="flex lg:hidden items-center justify-between px-4 py-3 border-b"
        style={{ backgroundColor: "#0C1210", borderColor: "#0A3D3D" }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="h-7 w-7 flex items-center justify-center font-mono text-xs font-bold"
            style={{
              backgroundColor: "#080808",
              border: "1px solid #00FFFF",
              color: "#00FFFF",
            }}
          >
            CE
          </div>
          <span className="font-mono text-sm font-bold tracking-wider text-[#EAFFFB]">
            CARDENHANCE <span className="text-[#00FF66] text-xs">{"// v3.0"}</span>
          </span>
        </div>
        <button
          type="button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-1.5 border text-[#00FFFF]"
          style={{ borderColor: "#0A3D3D", backgroundColor: "#080808" }}
        >
          {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* LEFT STICKY SIDEBAR (Strictly 7 Approved Routes, Zero Cloudflare/R2) */}
      <aside
        className={`${
          mobileMenuOpen ? "block" : "hidden"
        } lg:flex lg:flex-col lg:w-60 lg:shrink-0 lg:sticky lg:top-0 lg:h-screen justify-between border-r z-30`}
        style={{ backgroundColor: "#0C1210", borderColor: "#0A3D3D" }}
      >
        <div className="p-4 space-y-5">
          <div
            className="flex items-center justify-between pb-4 border-b"
            style={{ borderColor: "#0A3D3D" }}
          >
            <div className="flex items-center gap-2.5">
              <div
                className="h-8 w-8 flex items-center justify-center font-mono text-xs font-bold"
                style={{
                  backgroundColor: "#080808",
                  border: "1px solid #00FFFF",
                  color: "#00FFFF",
                  boxShadow: "0 0 12px rgba(0, 255, 255, 0.25)",
                }}
              >
                CE
              </div>
              <div>
                <div className="font-mono text-sm font-bold tracking-wider text-[#EAFFFB]">
                  CARDENHANCE
                </div>
                <div className="font-mono text-[10px] text-[#00FF66]">
                  {"// OBSIDIAN + RaGAN v3.0"}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-1">
            <div className="px-2 pb-1.5 font-mono text-[10px] tracking-widest text-[#00FF66]">
              {"// NAVIGATION (7 ROUTES)"}
            </div>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    setActivePage(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-left font-mono text-xs transition-all"
                  style={{
                    backgroundColor: isActive ? "rgba(0, 255, 255, 0.08)" : "transparent",
                    border: isActive ? "1px solid #00FFFF" : "1px solid transparent",
                    color: isActive ? "#00FFFF" : "#EAFFFB",
                    boxShadow: isActive ? "0 0 12px rgba(0, 255, 255, 0.14)" : "none",
                  }}
                >
                  <span className="flex items-center gap-2.5">
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{item.label}</span>
                  </span>
                  <span className="text-[10px] text-[#00FF66]">{"//"}{item.tag}</span>
                </button>
              );
            })}
          </div>

          {/* Live Service & Worker Telemetry */}
          <div
            className="p-3 space-y-2 font-mono text-[11px]"
            style={{ backgroundColor: "#080808", border: "1px solid #0A3D3D" }}
          >
            <div className="flex items-center justify-between text-[#00FF66]">
              <span>{"// INFERENCE STATUS"}</span>
              <button
                type="button"
                onClick={() => void checkHealth()}
                className="inline-flex items-center gap-1 text-[10px] text-[#00FFFF]"
                title="Refresh /api/health status"
              >
                <RefreshCw className={`h-3 w-3 ${checking ? "animate-spin" : ""}`} />
                {ready ? "GPU READY" : "LOCAL WORKER"}
              </button>
            </div>
            <div className="text-[10px] text-[#EAFFFB]/70 space-y-1">
              <div>• Browser Adaptive Engine: ON</div>
              <div>• 4-Way Duplex Auto-Orient: ON</div>
              <div>• FastAPI RaGAN Proxy: {ready ? "ONLINE" : "STANDBY"}</div>
              <div>• Max Upload: {formatBytes(service.maxBytes)}</div>
            </div>
          </div>
        </div>

        <div
          className="p-4 border-t font-mono text-[11px] space-y-1.5"
          style={{ borderColor: "#0A3D3D", backgroundColor: "#080808" }}
        >
          <div className="flex items-center justify-between">
            <span className="text-[#EAFFFB]/60">DEPLOYMENT</span>
            <span className="text-[#00FFFF]">VERCEL ONLY</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[#EAFFFB]/60">AVG SHARPNESS Δ</span>
            <span className="text-[#00FF66]">+{counts.avgDelta}%</span>
          </div>
        </div>
      </aside>

      {/* MAIN WORKSPACE */}
      <main className="flex-1 min-w-0 p-4 sm:p-6 space-y-5">
        {error && (
          <div
            className="flex items-center justify-between px-4 py-3 font-mono text-xs"
            style={{
              backgroundColor: "#0C1210",
              border: "1px solid #FF3366",
              color: "#EAFFFB",
            }}
            role="alert"
          >
            <span className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-[#FF3366]" />
              {error}
            </span>
            <button type="button" onClick={() => setError("")} className="text-[#EAFFFB]/60">
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* ================================================================= */}
        {/* ROUTE 02: STUDIO / BATCH EDITOR                                   */}
        {/* ================================================================= */}
        {activePage === "studio" && (
          <div className="space-y-4">
            {/* TOP BATCH TOOLBAR */}
            <div
              className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
              style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
            >
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: "#00FF66", boxShadow: "0 0 8px #00FF66" }}
                  />
                  <span className="font-mono text-sm font-bold tracking-wide text-[#EAFFFB]">
                    Batch editor
                  </span>
                </div>

                <span
                  className="px-3 py-1 font-mono text-xs"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#00FFFF",
                  }}
                >
                  {counts.total} scans · {counts.selected} selected · {counts.enhanced} enhanced
                </span>

                <button
                  type="button"
                  onClick={() => setAdaptiveMode((v) => !v)}
                  className="px-2.5 py-1 font-mono text-xs transition"
                  style={{
                    backgroundColor: adaptiveMode ? "rgba(0, 255, 102, 0.12)" : "#080808",
                    border: `1px solid ${adaptiveMode ? "#00FF66" : "#0A3D3D"}`,
                    color: adaptiveMode ? "#00FF66" : "#EAFFFB",
                  }}
                >
                  {"// ADAPTIVE AUTO: "}{adaptiveMode ? "ON" : "MANUAL"}
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
                <button
                  type="button"
                  onClick={() => batchInputRef.current?.click()}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 transition hover:border-[#00FFFF]"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  <Upload className="h-3.5 w-3.5 text-[#00FFFF]" />
                  Add scans
                </button>

                <button
                  type="button"
                  onClick={() => handleBatchRotate(-90)}
                  disabled={counts.selected === 0}
                  className="px-2 py-1.5 transition hover:border-[#00FFFF] disabled:opacity-40"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  ↺ 90°
                </button>
                <button
                  type="button"
                  onClick={() => handleBatchRotate(90)}
                  disabled={counts.selected === 0}
                  className="px-2 py-1.5 transition hover:border-[#00FFFF] disabled:opacity-40"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  ↻ 90°
                </button>
                <button
                  type="button"
                  onClick={() => handleBatchRotate(180)}
                  disabled={counts.selected === 0}
                  className="px-2 py-1.5 transition hover:border-[#00FFFF] disabled:opacity-40"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  180°
                </button>

                <button
                  type="button"
                  onClick={() => setCards((prev) => prev.map((c) => ({ ...c, selected: true })))}
                  className="px-2.5 py-1.5 transition hover:border-[#00FFFF]"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  Select all
                </button>

                <button
                  type="button"
                  onClick={() => setCards((prev) => prev.map((c) => ({ ...c, selected: false })))}
                  className="px-2.5 py-1.5 transition hover:border-[#00FFFF]"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #0A3D3D",
                    color: "#EAFFFB",
                  }}
                >
                  Clear
                </button>

                {processing ? (
                  <button
                    type="button"
                    onClick={() => requestRef.current?.abort()}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 font-bold"
                    style={{ backgroundColor: "#FF3366", color: "#080808" }}
                  >
                    <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                    Cancel
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={counts.selected === 0 || processing}
                    onClick={() => void runGpuRestoreOnSelected()}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 font-bold transition disabled:opacity-40"
                    style={{
                      backgroundColor: "#00FFFF",
                      color: "#080808",
                      boxShadow: "0 0 14px rgba(0, 255, 255, 0.35)",
                    }}
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    {processing ? "Processing..." : `Run on ${counts.selected} selected`}
                  </button>
                )}
              </div>
            </div>

            {/* STUDIO 2-COLUMN WORKSPACE */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
              {/* LEFT INSPECTOR RAIL: ENHANCEMENT, GPU TILING & EXPORT */}
              <aside
                className="xl:col-span-4 2xl:col-span-3 p-4 space-y-5 xl:sticky xl:top-4"
                style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
              >
                <div className="space-y-4">
                  <div
                    className="flex items-center justify-between border-b pb-2"
                    style={{ borderColor: "#0A3D3D" }}
                  >
                    <span className="font-mono text-xs font-bold tracking-wider text-[#00FF66]">
                      {"// 01 ENHANCEMENT"}
                    </span>
                    <span className="font-mono text-[10px] text-[#00FFFF]">
                      {adaptiveMode ? "ADAPTIVE BIAS" : "MANUAL"}
                    </span>
                  </div>

                  {/* Scale */}
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Scale</span>
                      <span className="text-[#00FFFF]">{scale.toFixed(1)}x</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="4"
                      step="0.5"
                      value={scale}
                      onChange={(e) => setScale(Number(e.target.value))}
                      className="w-full accent-[#00FFFF]"
                    />
                    <p className="font-mono text-[10px] text-[#EAFFFB]/60">
                      Catmull-Rom bicubic resample (long edge capped at 2200px).
                    </p>
                  </div>

                  {/* Descratch + Chrome Checkbox */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Descratch</span>
                      <span className="text-[#00FFFF]">{descratch.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={descratch}
                      onChange={(e) => setDescratch(Number(e.target.value))}
                      className="w-full accent-[#00FFFF]"
                    />
                    <p className="font-mono text-[10px] text-[#EAFFFB]/60">
                      Multi-scale [3..16px] 1D CIS streak &amp; dust removal before denoise.
                    </p>
                    <label className="flex items-start gap-2 pt-1 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={onlyChromeDescratch}
                        onChange={(e) => setOnlyChromeDescratch(e.target.checked)}
                        className="mt-0.5 accent-[#00FFFF]"
                      />
                      <span className="font-mono text-[11px] text-[#EAFFFB]/85 leading-snug">
                        Only on scans detected as chrome/foil (specular-highlight heuristic)
                      </span>
                    </label>
                  </div>

                  {/* Denoise */}
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Denoise</span>
                      <span className="text-[#00FFFF]">{denoise.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={denoise}
                      onChange={(e) => setDenoise(Number(e.target.value))}
                      className="w-full accent-[#00FFFF]"
                    />
                    <p className="font-mono text-[10px] text-[#EAFFFB]/60">
                      Edge-aware 3×3 box average blend over original pixels.
                    </p>
                  </div>

                  {/* Sharpen */}
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Sharpen</span>
                      <span className="text-[#00FFFF]">{sharpen.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={sharpen}
                      onChange={(e) => setSharpen(Number(e.target.value))}
                      className="w-full accent-[#00FFFF]"
                    />
                    <p className="font-mono text-[10px] text-[#EAFFFB]/60">
                      Highlight-safe unsharp mask updating measured sharpness delta.
                    </p>
                  </div>

                  {/* Contrast */}
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Contrast</span>
                      <span className="text-[#00FFFF]">{contrast.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="0.35"
                      step="0.01"
                      value={contrast}
                      onChange={(e) => setContrast(Number(e.target.value))}
                      className="w-full accent-[#00FFFF]"
                    />
                    <p className="font-mono text-[10px] text-[#EAFFFB]/60">
                      Foil-safe mid-grey S-curve + CIS black-point &amp; glare recovery.
                    </p>
                  </div>

                  <label className="flex items-center justify-between pt-1 font-mono text-xs cursor-pointer">
                    <span>Auto-split 2×2 / 3×3 sheets</span>
                    <input
                      type="checkbox"
                      checked={splitSheets}
                      onChange={(e) => setSplitSheets(e.target.checked)}
                      className="accent-[#00FFFF]"
                    />
                  </label>
                </div>

                {/* GPU TILED INFERENCE SECTION (Preserves d78fb64 FastAPI Proxy Controls) */}
                <div className="space-y-3 pt-3 border-t font-mono" style={{ borderColor: "#0A3D3D" }}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-[#00FF66]">{"// 02 GPU RaGAN PROXY"}</span>
                    <span className="text-[10px] text-[#00FFFF]">
                      {ready ? service.generator?.toUpperCase() || "RRDB" : "STANDBY"}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <label htmlFor="tile-size" className="block text-[10px] text-[#EAFFFB]/70 mb-1">
                        Tile size
                      </label>
                      <select
                        id="tile-size"
                        value={tileSize}
                        onChange={(e) => {
                          const next = Number(e.target.value);
                          setTileSize(next);
                          if (next > 0 && tileOverlap >= next) setTileOverlap(Math.max(0, next - 1));
                        }}
                        className="w-full px-2 py-1 text-xs"
                        style={{
                          backgroundColor: "#080808",
                          border: "1px solid #0A3D3D",
                          color: "#EAFFFB",
                        }}
                      >
                        <option value="0">Full frame</option>
                        <option value="256">256 px</option>
                        <option value="512">512 px</option>
                        <option value="1024">1024 px</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="tile-overlap" className="block text-[10px] text-[#EAFFFB]/70 mb-1">
                        Overlap ({tileSize === 0 ? "Off" : `${tileOverlap}px`})
                      </label>
                      <input
                        id="tile-overlap"
                        type="range"
                        min="0"
                        max="256"
                        step="8"
                        value={tileOverlap}
                        disabled={tileSize === 0}
                        onChange={(e) => setTileOverlap(Number(e.target.value))}
                        className="w-full accent-[#00FFFF] mt-1.5"
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[11px] text-[#EAFFFB]/80">CUDA Mixed Precision (AMP)</span>
                    <input
                      type="checkbox"
                      checked={amp}
                      onChange={(e) => setAmp(e.target.checked)}
                      className="accent-[#00FFFF]"
                    />
                  </div>
                </div>

                {/* EXPORT SECTION */}
                <div className="space-y-2.5 pt-3 border-t" style={{ borderColor: "#0A3D3D" }}>
                  <div className="font-mono text-xs font-bold tracking-wider text-[#00FF66]">
                    {"// 03 EXPORT"}
                  </div>
                  <button
                    type="button"
                    onClick={handleDownloadEnhancedPngs}
                    disabled={counts.selected === 0}
                    className="w-full flex items-center justify-center gap-2 py-2 px-3 font-mono text-xs font-semibold transition disabled:opacity-40"
                    style={{
                      backgroundColor: "#080808",
                      border: "1px solid #00FFFF",
                      color: "#00FFFF",
                    }}
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download enhanced PNGs
                  </button>
                  <button
                    type="button"
                    onClick={handleExportAuditJson}
                    disabled={cards.length === 0}
                    className="w-full flex items-center justify-center gap-2 py-2 px-3 font-mono text-xs transition disabled:opacity-40"
                    style={{
                      backgroundColor: "#080808",
                      border: "1px solid #0A3D3D",
                      color: "#EAFFFB",
                    }}
                  >
                    <FileJson className="h-3.5 w-3.5 text-[#00FF66]" />
                    Export audit JSON
                  </button>
                </div>
              </aside>

              {/* RIGHT MIXED PORTRAIT + LANDSCAPE CARD GRID OR COMPACT DROPZONE */}
              <div className="xl:col-span-8 2xl:col-span-9">
                {cards.length === 0 ? (
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={onDrop}
                    className="w-full p-12 flex flex-col items-center justify-center text-center transition"
                    style={{
                      backgroundColor: "#0C1210",
                      border: `1px dashed ${dragging ? "#00FFFF" : "#0A3D3D"}`,
                      boxShadow: dragging ? "0 0 24px rgba(0, 255, 255, 0.2)" : "none",
                    }}
                  >
                    <div
                      className="h-12 w-12 flex items-center justify-center mb-3"
                      style={{
                        backgroundColor: "#080808",
                        border: "1px solid #00FFFF",
                        color: "#00FFFF",
                      }}
                    >
                      <Upload className="h-6 w-6" />
                    </div>
                    <span className="font-mono text-sm font-bold text-[#EAFFFB]">
                      Drop raw card scans or 2×2 / 3×3 flatbed sheets here
                    </span>
                    <span className="font-mono text-xs text-[#EAFFFB]/60 mt-1">
                      JPEG, PNG, or WebP up to {formatBytes(service.maxBytes)} · Auto-rotates 0°/90°/180°/270° &amp; removes CIS streaks
                    </span>
                    <span
                      className="mt-4 px-4 py-2 font-mono text-xs font-bold"
                      style={{ backgroundColor: "#00FFFF", color: "#080808" }}
                    >
                      Choose card scans
                    </span>
                  </button>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {cards.map((card) => {
                      const isLandscape = card.audit.isLandscapeCard;
                      const p = card.audit.appliedParams;
                      return (
                        <div
                          key={card.id}
                          className={`flex flex-col justify-between p-3 transition-all ${
                            isLandscape ? "sm:col-span-2" : "col-span-1"
                          } ${card.selected ? "opacity-100" : "opacity-75 hover:opacity-95"}`}
                          style={{
                            backgroundColor: "#0C1210",
                            border: card.selected ? "1px solid #00FFFF" : "1px solid #0A3D3D",
                            boxShadow: card.selected
                              ? "0 0 16px rgba(0, 255, 255, 0.16)"
                              : "none",
                          }}
                        >
                          <div className="flex items-center justify-between gap-1.5 mb-2.5">
                            <div className="flex items-center gap-2 min-w-0">
                              <button
                                type="button"
                                onClick={() =>
                                  setCards((prev) =>
                                    prev.map((c) =>
                                      c.id === card.id ? { ...c, selected: !c.selected } : c
                                    )
                                  )
                                }
                                className="h-5 w-5 shrink-0 flex items-center justify-center"
                                style={{
                                  backgroundColor: card.selected ? "#00FFFF" : "#080808",
                                  border: `1px solid ${card.selected ? "#00FFFF" : "#0A3D3D"}`,
                                  color: "#080808",
                                }}
                              >
                                {card.selected && <Check className="h-3.5 w-3.5 stroke-[3]" />}
                              </button>
                              <span
                                className="truncate px-2 py-0.5 font-mono text-[10px] font-semibold"
                                style={{
                                  backgroundColor: "#080808",
                                  border: `1px solid ${
                                    card.audit.isChromeFoil ? "#00FFFF" : "#0A3D3D"
                                  }`,
                                  color: card.audit.isChromeFoil ? "#00FFFF" : "#00FF66",
                                }}
                              >
                                {card.audit.profile}
                              </span>
                            </div>

                            <div className="flex items-center gap-1 font-mono text-[10px]">
                              <button
                                type="button"
                                onClick={() => handleQuickRotate(card.id, -90)}
                                className="px-1.5 py-0.5 hover:border-[#00FFFF]"
                                style={{ backgroundColor: "#080808", border: "1px solid #0A3D3D" }}
                              >
                                ↺
                              </button>
                              <button
                                type="button"
                                onClick={() => handleQuickRotate(card.id, 90)}
                                className="px-1.5 py-0.5 hover:border-[#00FFFF]"
                                style={{ backgroundColor: "#080808", border: "1px solid #0A3D3D" }}
                              >
                                ↻
                              </button>
                              <button
                                type="button"
                                onClick={() => handleQuickRotate(card.id, 180)}
                                className="px-1.5 py-0.5 hover:border-[#00FFFF]"
                                style={{ backgroundColor: "#080808", border: "1px solid #0A3D3D" }}
                              >
                                180°
                              </button>
                            </div>
                          </div>

                          <div
                            onClick={() => {
                              setActiveCardId(card.id);
                              setActivePage("detail");
                            }}
                            className="relative flex items-center justify-center overflow-hidden cursor-pointer group py-3 px-2"
                            style={{
                              backgroundColor: "#080808",
                              border: "1px solid #0A3D3D",
                              minHeight: "260px",
                              maxHeight: "300px",
                            }}
                          >
                            <img
                              src={card.processedUrl}
                              alt={card.filename}
                              className="max-h-[265px] max-w-full w-auto h-auto object-contain"
                            />
                          </div>

                          <div className="mt-2.5 space-y-1 font-mono">
                            <div className="flex items-center justify-between gap-2 text-xs">
                              <span className="truncate font-semibold text-[#EAFFFB]">
                                {card.filename}
                              </span>
                              <span className="shrink-0 font-bold text-[#00FF66]">
                                +{card.audit.sharpnessDeltaPct.toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex flex-wrap items-center justify-between gap-1 text-[10px] text-[#EAFFFB]/65">
                              <span>
                                {card.audit.outputSize.width} × {card.audit.outputSize.height}
                                {isLandscape ? " · landscape" : ""}
                              </span>
                              <span className="text-[#00FFFF]">
                                rot {card.audit.appliedRotationDeg}° · {card.audit.cisStreaksRemoved} streaks
                              </span>
                            </div>
                            <div
                              className="pt-1 flex items-center justify-between text-[10px] border-t"
                              style={{ borderColor: "#0A3D3D", color: "#00FF66" }}
                            >
                              <span>
                                d{p.descratch.toFixed(2)} n{p.denoise.toFixed(2)} s{p.sharpen.toFixed(2)} c{p.contrast.toFixed(2)}
                              </span>
                              <span className="text-[#EAFFFB]/50">
                                {card.gpuEnhanced ? "RaGAN GPU" : card.audit.orientationSource}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* ROUTE 01: DASHBOARD                                               */}
        {/* ================================================================= */}
        {activePage === "dashboard" && (
          <div className="space-y-5">
            <div
              className="p-5 space-y-3"
              style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
            >
              <div className="font-mono text-xs text-[#00FF66]">
                {"// 01 RELATIVISTIC SURFACE RESTORATION & ADAPTIVE BATCH ENGINE"}
              </div>
              <h1 className="text-2xl font-bold text-[#EAFFFB]">
                Keep the reflection. Lose the scanner damage.
              </h1>
              <p className="text-sm text-[#EAFFFB]/75 max-w-3xl">
                Combines browser-native 4-way duplex orientation, multi-width CIS streak removal, and corner-safe die-cut cropping with your private FastAPI PyTorch RaGAN GPU endpoint.
              </p>
              <div className="flex flex-wrap gap-3 pt-2 font-mono text-xs">
                <button
                  type="button"
                  onClick={() => setActivePage("studio")}
                  className="px-4 py-2 font-bold"
                  style={{ backgroundColor: "#00FFFF", color: "#080808" }}
                >
                  Open Studio
                </button>
                <button
                  type="button"
                  onClick={() => setActivePage("settings")}
                  className="px-4 py-2"
                  style={{
                    backgroundColor: "#080808",
                    border: "1px solid #00FFFF",
                    color: "#00FFFF",
                  }}
                >
                  GPU Endpoint Config
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* ROUTE 04: CARD DETAIL (3D TILT + SPLIT COMPARISON VIEW)           */}
        {/* ================================================================= */}
        {activePage === "detail" && (
          <div className="space-y-5">
            {!activeCard ? (
              <div
                className="p-8 text-center font-mono text-xs"
                style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
              >
                Upload a scan in Studio first to inspect in 3D &amp; split comparison view.
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                <div
                  className="lg:col-span-7 p-5 space-y-4"
                  style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
                >
                  <div className="flex items-center justify-between font-mono text-xs">
                    <span className="text-[#00FF66]">{"// 04A INTERACTIVE 3D HOLOGRAPHIC STAGE"}</span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setCompareBefore((v) => !v)}
                        className="px-2.5 py-1"
                        style={{
                          backgroundColor: "#080808",
                          border: "1px solid #00FFFF",
                          color: "#00FFFF",
                        }}
                      >
                        {compareBefore ? "Showing: RAW" : "Showing: RESTORED"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setHoloFoilEnabled((v) => !v)}
                        className="px-2.5 py-1"
                        style={{
                          backgroundColor: "#080808",
                          border: "1px solid #00FF66",
                          color: "#00FF66",
                        }}
                      >
                        Holo: {holoFoilEnabled ? "ON" : "OFF"}
                      </button>
                    </div>
                  </div>

                  <div
                    onMouseMove={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const nx = (e.clientX - rect.left) / rect.width;
                      const ny = (e.clientY - rect.top) / rect.height;
                      setTilt({
                        rx: (0.5 - ny) * 26,
                        ry: (nx - 0.5) * 32,
                        glareX: nx * 100,
                        glareY: ny * 100,
                      });
                    }}
                    className="flex items-center justify-center py-8"
                    style={{ perspective: "1100px" }}
                  >
                    <div
                      className="relative rounded-2xl overflow-hidden"
                      style={{
                        transform: `rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg)`,
                        border: "2px solid #00FFFF",
                        maxWidth: "320px",
                      }}
                    >
                      <img
                        src={compareBefore ? activeCard.rawUrl : activeCard.processedUrl}
                        alt={activeCard.filename}
                        className="w-full h-auto block"
                      />
                      {holoFoilEnabled && !compareBefore && (
                        <div
                          className="absolute inset-0 pointer-events-none mix-blend-screen"
                          style={{
                            background: `radial-gradient(circle at ${tilt.glareX}% ${tilt.glareY}%, rgba(0,255,255,0.32), rgba(0,255,102,0.18) 40%, transparent 80%)`,
                          }}
                        />
                      )}
                    </div>
                  </div>
                </div>

                <div
                  className="lg:col-span-5 p-5 space-y-4"
                  style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
                >
                  <div className="font-mono text-xs text-[#00FF66]">
                    {"// 04B BEFORE / AFTER DIVIDER COMPARISON"}
                  </div>
                  <ComparisonView
                    sourceUrl={activeCard.rawUrl}
                    resultUrl={activeCard.processedUrl}
                    sourceName={activeCard.filename}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* ROUTES 03, 05, 06, 07: LIBRARY, AUDIT, ASSISTANT, SETTINGS        */}
        {/* ================================================================= */}
        {(activePage === "library" ||
          activePage === "audit" ||
          activePage === "assistant" ||
          activePage === "settings") && (
          <div
            className="p-5 space-y-4 font-mono"
            style={{ backgroundColor: "#0C1210", border: "1px solid #0A3D3D" }}
          >
            <div className="text-xs text-[#00FF66]">
                {"// "}{activePage.toUpperCase()} {" MODULE — VERCEL + EXTERNAL FASTAPI GPU"}
            </div>
            <p className="text-xs text-[#EAFFFB]/80 leading-relaxed">
              Configure these server-only environment variables in Vercel to connect your external FastAPI/PyTorch checkpoint service (`inference/server.py`) alongside the in-browser Adaptive Restoration Worker:
            </p>
            <pre
              className="p-3 text-xs overflow-x-auto"
              style={{ backgroundColor: "#080808", border: "1px solid #0A3D3D", color: "#00FFFF" }}
            >
              <code>{`RESTORATION_API_URL=https://your-gpu-host.example
RESTORATION_API_KEY=your-long-random-secret
MAX_IMAGE_BYTES=4000000`}</code>
            </pre>
            {dimensions && (
              <div className="text-xs text-[#00FF66]">
                Last loaded scan dimensions: {dimensions.width} × {dimensions.height} px
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
