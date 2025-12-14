"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Sliders,
  Palette,
  Eye,
  CropIcon,
  Image as ImageIcon,
  Wand2,
  EyeOff,
  Grid3x3,
  Download,
  RotateCcw,
  Undo2,
  Redo2,
  Ruler,
  Sun,
  CircleIcon,
  Zap,
  Cloud,
  Square,
  Droplet,
  Thermometer,
  Compass,
  Maximize,
  Move,
  RotateCw,
} from "lucide-react";

// Type definitions
interface State {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  temp: number;
  tint: number;
  saturation: number;
  vibrance: number;
  rotate: number;
  scale: number;
  perspectiveX: number;
  perspectiveY: number;
  distortion: number;
  vignette: number;
  vignetteMid: number;
  hsl: {
    h: number[];
    s: number[];
    l: number[];
  };
  activeHSLChannel: number;
  cropAspect: number;
  gridType: string;
}

interface UniformLocations {
  [key: string]: WebGLUniformLocation | null;
}

const DEFAULT_IMG =
  "https://images.unsplash.com/photo-1493863641943-9b68992a8d07?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80";

const DEFAULT_STATE: State = {
  exposure: 0,
  contrast: 1,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  temp: 0,
  tint: 0,
  saturation: 0,
  vibrance: 0,
  rotate: 0,
  scale: 1,
  perspectiveX: 0,
  perspectiveY: 0,
  distortion: 0,
  vignette: 0,
  vignetteMid: 0.5,
  hsl: {
    h: Array(8).fill(0),
    s: Array(8).fill(0),
    l: Array(8).fill(0),
  },
  activeHSLChannel: 0,
  cropAspect: 0,
  gridType: "none",
};

const vertexShaderSource = `
attribute vec2 a_position;
varying vec2 v_texCoord;
uniform float u_imgAspect;
uniform float u_viewAspect;
uniform float u_rotate;
uniform float u_scale;
uniform vec2 u_perspective;

void main() {
    v_texCoord = (a_position + 1.0) / 2.0;
    v_texCoord.y = 1.0 - v_texCoord.y;
    vec2 pos = a_position;

    pos.x *= u_imgAspect;

    float s = sin(u_rotate);
    float c = cos(u_rotate);
    pos = mat2(c, -s, s, c) * pos;

    float w = 1.0 + (pos.x * u_perspective.x * 0.5) + (pos.y * -u_perspective.y * 0.5);
    w = max(w, 0.05);

    pos *= u_scale;
    pos.x /= u_viewAspect;

    gl_Position = vec4(pos.x, pos.y, 0.0, w);
}`;

const fragmentShaderSource = `
precision mediump float;
varying vec2 v_texCoord;
uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;
uniform float u_temp;
uniform float u_tint;
uniform float u_saturation;
uniform float u_vibrance;
uniform float u_distortion;
uniform float u_vignette;
uniform float u_vignetteMid;
uniform float u_hsl_h[8];
uniform float u_hsl_s[8];
uniform float u_hsl_l[8];

float getLuma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec3 rgb2hsl(vec3 c) {
    float maxC = max(max(c.r, c.g), c.b), minC = min(min(c.r, c.g), c.b);
    float l = (maxC + minC) / 2.0;
    if (maxC == minC) return vec3(0.0, 0.0, l);
    float d = maxC - minC;
    float s = l > 0.5 ? d / max(2.0 - maxC - minC, 1e-4) : d / max(maxC + minC, 1e-4);
    float h = (maxC == c.r) ? (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0) :
              (maxC == c.g) ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0;
    return vec3(clamp(h/6.0, 0.0, 1.0), clamp(s, 0.0, 1.0), clamp(l, 0.0, 1.0));
}

float hue2rgb(float p, float q, float t) {
    if(t < 0.0) t += 1.0;
    if(t > 1.0) t -= 1.0;
    if(t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if(t < 1.0/2.0) return q;
    if(t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}

vec3 hsl2rgb(vec3 hsl) {
    float h = hsl.x, s = hsl.y, l = hsl.z;
    if(s == 0.0) return vec3(l);
    float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
    float p = 2.0 * l - q;
    return vec3(hue2rgb(p, q, h + 1.0/3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0/3.0));
}

float getHueWeight(float h, float c, float w) {
    float d = abs(h - c);
    if(d > 0.5) d = 1.0 - d;
    return smoothstep(w, 0.0, d);
}

void main() {
    vec2 uv = v_texCoord;
    vec2 center = vec2(0.5);
    float r2 = dot(uv - center, uv - center);
    uv = center + (uv - center) * (1.0 + u_distortion * r2);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(0.0);
        return;
    }

    vec3 rgb = texture2D(u_image, uv).rgb;
    rgb.r += u_temp + u_tint;
    rgb.g += u_tint;
    rgb.b += -u_temp + u_tint;
    rgb *= pow(2.0, u_exposure);
    rgb = (rgb - 0.5) * u_contrast + 0.5;

    float l = getLuma(rgb);
    rgb += u_shadows * (1.0-l)*(1.0-l)*0.5 + u_highlights * l*l*0.5;
    rgb += u_blacks * (1.0 - smoothstep(0.0, 0.4, l))*0.25 + u_whites * smoothstep(0.6, 1.0, l)*0.25;
    rgb = clamp(rgb, 0.0, 1.0);

    vec3 hsl = rgb2hsl(rgb);
    float w[8];
    w[0]=getHueWeight(hsl.x,0.,.1)+getHueWeight(hsl.x,1.,.1);
    w[1]=getHueWeight(hsl.x,.08,.1);
    w[2]=getHueWeight(hsl.x,.16,.1);
    w[3]=getHueWeight(hsl.x,.33,.15);
    w[4]=getHueWeight(hsl.x,.5,.1);
    w[5]=getHueWeight(hsl.x,.6,.15);
    w[6]=getHueWeight(hsl.x,.75,.1);
    w[7]=getHueWeight(hsl.x,.83,.1);

    float hS=0., sS=0., lS=0.;
    for(int i=0;i<8;i++) {
        hS+=w[i]*u_hsl_h[i];
        sS+=w[i]*u_hsl_s[i];
        lS+=w[i]*u_hsl_l[i];
    }

    hsl.x+=hS;
    hsl.y*=(1.+sS)*(1.+u_saturation);
    hsl.z*=(1.+lS);
    hsl.y*=(1.+u_vibrance*(1.-hsl.y));
    rgb = hsl2rgb(hsl);

    float vig = smoothstep(u_vignetteMid - 0.2, u_vignetteMid + 0.6, distance(gl_FragCoord.xy / u_resolution, vec2(0.5)));
    rgb = mix(rgb, rgb * (1.0 - u_vignette), vig);
    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}`;

export default function AutoHDRStudio() {
  const [state, setState] = useState<State>(DEFAULT_STATE);
  const [activeTab, setActiveTab] = useState("basic");
  const [undoStack, setUndoStack] = useState<State[]>([]);
  const [redoStack, setRedoStack] = useState<State[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showToast, setShowToast] = useState({ show: false, message: "" });
  const [isStraightening, setIsStraightening] = useState(false);
  const [straightenStart, setStraightenStart] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [showGridMenu, setShowGridMenu] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const histogramCanvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const uniformLocsRef = useRef<UniformLocations>({});
  const imgAspectRef = useRef(1);
  const animationFrameRef = useRef<number>();

  // Initialize WebGL
  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      preserveDrawingBuffer: true,
      alpha: true,
    });

    if (!gl) {
      console.error("WebGL not supported");
      return;
    }

    glRef.current = gl;

    // Create shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(
      gl,
      gl.FRAGMENT_SHADER,
      fragmentShaderSource,
    );

    if (!vertexShader || !fragmentShader) return;

    const program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) return;

    programRef.current = program;

    // Set up geometry
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW,
    );

    const aPos = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Get uniform locations
    const uniforms = [
      "u_imgAspect",
      "u_viewAspect",
      "u_rotate",
      "u_scale",
      "u_perspective",
      "u_exposure",
      "u_contrast",
      "u_highlights",
      "u_shadows",
      "u_whites",
      "u_blacks",
      "u_temp",
      "u_tint",
      "u_saturation",
      "u_vibrance",
      "u_distortion",
      "u_vignette",
      "u_vignetteMid",
      "u_resolution",
    ];

    const locs: UniformLocations = {};
    uniforms.forEach((name) => {
      locs[name] = gl.getUniformLocation(program, name);
    });

    locs.hsl_h = gl.getUniformLocation(program, "u_hsl_h");
    locs.hsl_s = gl.getUniformLocation(program, "u_hsl_s");
    locs.hsl_l = gl.getUniformLocation(program, "u_hsl_l");

    uniformLocsRef.current = locs;
  }, []);

  const createShader = (
    gl: WebGLRenderingContext,
    type: number,
    source: string,
  ) => {
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  };

  const createProgram = (
    gl: WebGLRenderingContext,
    vertexShader: WebGLShader,
    fragmentShader: WebGLShader,
  ) => {
    const program = gl.createProgram();
    if (!program) return null;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program link error:", gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }

    return program;
  };

  const createTexture = (gl: WebGLRenderingContext, img: HTMLImageElement) => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);

    textureRef.current = texture;
  };

  const loadImage = (url: string) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;

    img.onload = () => {
      imgAspectRef.current = img.width / img.height;
      if (glRef.current) {
        createTexture(glRef.current, img);
      }
      setIsLoading(false);
      resizeViewport();
    };

    img.onerror = () => {
      console.error("Error loading image");
      setIsLoading(false);
    };
  };

  const resizeViewport = () => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    const viewport = wrapper?.parentElement;

    if (!canvas || !wrapper || !viewport) return;

    const maxWidth = viewport.clientWidth - 40;
    const maxHeight = viewport.clientHeight - 40;
    const aspectRatio = state.cropAspect || imgAspectRef.current;

    let width = maxWidth;
    let height = width / aspectRatio;

    if (height > maxHeight) {
      height = maxHeight;
      width = height * aspectRatio;
    }

    canvas.width = Math.floor(width);
    canvas.height = Math.floor(height);

    if (overlayCanvasRef.current) {
      overlayCanvasRef.current.width = width;
      overlayCanvasRef.current.height = height;
    }

    wrapper.style.width = `${width}px`;
    wrapper.style.height = `${height}px`;

    if (glRef.current) {
      glRef.current.viewport(0, 0, width, height);
    }
  };

  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const texture = textureRef.current;
    const canvas = canvasRef.current;

    if (!gl || !program || !texture || !canvas) return;

    resizeViewport();

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    const viewAR = canvas.width / canvas.height;
    const angle = (state.rotate * Math.PI) / 180;

    // Calculate transform for auto-crop
    const vCorners = [
      { x: -1, y: -1 },
      { x: 1, y: -1 },
      { x: 1, y: 1 },
      { x: -1, y: 1 },
    ];

    const poly = vCorners.map((corner) => {
      let x = corner.x * imgAspectRef.current;
      let y = corner.y;

      const rx = x * Math.cos(angle) - y * Math.sin(angle);
      const ry = x * Math.sin(angle) + y * Math.cos(angle);

      let pZ =
        1.0 + rx * state.perspectiveX * 0.5 + ry * -state.perspectiveY * 0.5;
      pZ = Math.max(pZ, 0.05);

      return { x: rx / pZ / viewAR, y: ry / pZ };
    });

    let maxReq = 0;
    for (let i = 0; i < 4; i++) {
      const p1 = poly[i];
      const p2 = poly[(i + 1) % 4];

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      let nx = -dy;
      let ny = dx;
      const len = Math.sqrt(nx * nx + ny * ny);

      if (len > 0) {
        nx /= len;
        ny /= len;
      }

      let dist = nx * p1.x + ny * p1.y;
      if (dist < 0) {
        nx = -nx;
        ny = -ny;
        dist = -dist;
      }

      const boxExt = Math.abs(nx) + Math.abs(ny);
      if (dist > 0.0001) {
        const s = boxExt / dist;
        if (s > maxReq) maxReq = s;
      } else {
        maxReq = 100;
      }
    }

    if (!Number.isFinite(maxReq) || maxReq < 0.1) maxReq = 1.0;
    const finalScale = Math.max(state.scale, 1.0) * maxReq * 1.02;

    // Set uniforms
    const locs = uniformLocsRef.current;
    gl.uniform1f(locs.u_imgAspect, imgAspectRef.current);
    gl.uniform1f(locs.u_viewAspect, viewAR);
    gl.uniform1f(locs.u_rotate, angle);
    gl.uniform1f(locs.u_scale, finalScale);
    gl.uniform2f(locs.u_perspective, state.perspectiveX, state.perspectiveY);
    gl.uniform2f(locs.u_resolution, canvas.width, canvas.height);
    gl.uniform1f(locs.u_exposure, state.exposure);
    gl.uniform1f(locs.u_contrast, state.contrast);
    gl.uniform1f(locs.u_highlights, state.highlights);
    gl.uniform1f(locs.u_shadows, state.shadows);
    gl.uniform1f(locs.u_whites, state.whites);
    gl.uniform1f(locs.u_blacks, state.blacks);
    gl.uniform1f(locs.u_temp, state.temp);
    gl.uniform1f(locs.u_tint, state.tint);
    gl.uniform1f(locs.u_saturation, state.saturation);
    gl.uniform1f(locs.u_vibrance, state.vibrance);
    gl.uniform1fv(locs.hsl_h, state.hsl.h);
    gl.uniform1fv(locs.hsl_s, state.hsl.s);
    gl.uniform1fv(locs.hsl_l, state.hsl.l);
    gl.uniform1f(locs.u_distortion, state.distortion);
    gl.uniform1f(locs.u_vignette, state.vignette);
    gl.uniform1f(locs.u_vignetteMid, state.vignetteMid);

    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);

    // Draw histogram occasionally
    if (performance.now() % 50 < 16) {
      drawHistogram();
    }
  }, [state]);

  const drawHistogram = () => {
    const gl = glRef.current;
    const canvas = canvasRef.current;
    const histCanvas = histogramCanvasRef.current;

    if (!gl || !canvas || !histCanvas) return;

    const ctx = histCanvas.getContext("2d");
    if (!ctx) return;

    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    const r = new Uint32Array(256);
    const g = new Uint32Array(256);
    const b = new Uint32Array(256);

    for (let i = 0; i < pixels.length; i += 64) {
      r[pixels[i]]++;
      g[pixels[i + 1]]++;
      b[pixels[i + 2]]++;
    }

    const max = Math.max(Math.max(...r), Math.max(...g), Math.max(...b)) || 1;
    const cw = histCanvas.width;
    const ch = histCanvas.height;

    ctx.clearRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = "screen";

    const colors = [
      "rgba(255,50,50,0.8)",
      "rgba(50,255,50,0.8)",
      "rgba(50,50,255,0.8)",
    ];
    [r, g, b].forEach((bin, i) => {
      ctx.fillStyle = colors[i];
      ctx.beginPath();
      ctx.moveTo(0, ch);
      for (let j = 0; j < 256; j++) {
        ctx.lineTo((j / 255) * cw, ch - (bin[j] / max) * ch);
      }
      ctx.lineTo(cw, ch);
      ctx.fill();
    });
  };

  const animate = useCallback(() => {
    render();
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [render]);

  const saveState = () => {
    const newState = JSON.parse(JSON.stringify(state));
    setUndoStack((prev) => [...prev.slice(-19), newState]);
    setRedoStack([]);
  };

  const handleUndo = () => {
    if (undoStack.length > 1) {
      const newUndoStack = [...undoStack];
      const currentState = newUndoStack.pop()!;
      setRedoStack((prev) => [...prev, currentState]);
      setState(newUndoStack[newUndoStack.length - 1]);
      setUndoStack(newUndoStack);
      displayToast("Undo");
    }
  };

  const handleRedo = () => {
    if (redoStack.length > 0) {
      const newRedoStack = [...redoStack];
      const stateToRedo = newRedoStack.pop()!;
      setUndoStack((prev) => [...prev, stateToRedo]);
      setState(stateToRedo);
      setRedoStack(newRedoStack);
      displayToast("Redo");
    }
  };

  const handleReset = () => {
    setState(DEFAULT_STATE);
    saveState();
    displayToast("Reset Complete");
  };

  const displayToast = (message: string) => {
    setShowToast({ show: true, message });
    setTimeout(() => setShowToast({ show: false, message: "" }), 2000);
  };

  // Auto-tone functionality
  const analyzeImageHistogram = (): {
    brightness: number;
    contrast: number;
    highlights: number;
    shadows: number;
  } => {
    const gl = glRef.current;
    const canvas = canvasRef.current;

    if (!gl || !canvas)
      return { brightness: 0, contrast: 1, highlights: 0, shadows: 0 };

    // Read pixels
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Build luminance histogram
    const histogram = new Uint32Array(256);
    let totalPixels = 0;

    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const luma = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      histogram[luma]++;
      totalPixels++;
    }

    // Find percentiles
    let cumulative = 0;
    let p5 = 0,
      p50 = 0,
      p95 = 0;

    for (let i = 0; i < 256; i++) {
      cumulative += histogram[i];
      const percentile = cumulative / totalPixels;

      if (percentile >= 0.05 && p5 === 0) p5 = i;
      if (percentile >= 0.5 && p50 === 0) p50 = i;
      if (percentile >= 0.95 && p95 === 0) p95 = i;
    }

    // Calculate adjustments
    const targetMidpoint = 128;
    const brightness = (targetMidpoint - p50) / 128;

    const currentRange = p95 - p5;
    const targetRange = 200; // Aim for good contrast
    const contrast = currentRange > 0 ? targetRange / currentRange : 1;

    // Analyze highlights and shadows
    let shadowSum = 0,
      shadowCount = 0;
    let highlightSum = 0,
      highlightCount = 0;

    for (let i = 0; i < 64; i++) {
      shadowSum += i * histogram[i];
      shadowCount += histogram[i];
    }

    for (let i = 192; i < 256; i++) {
      highlightSum += i * histogram[i];
      highlightCount += histogram[i];
    }

    const avgShadow = shadowCount > 0 ? shadowSum / shadowCount : 32;
    const avgHighlight =
      highlightCount > 0 ? highlightSum / highlightCount : 224;

    const shadows = (32 - avgShadow) / 64;
    const highlights = (224 - avgHighlight) / 64;

    return { brightness, contrast, highlights, shadows };
  };

  const handleAutoTone = () => {
    render(); // Ensure we have current pixels
    const analysis = analyzeImageHistogram();

    setState((prev) => ({
      ...prev,
      exposure: Math.max(-2, Math.min(2, analysis.brightness * 0.5)),
      contrast: Math.max(0.5, Math.min(1.5, analysis.contrast)),
      highlights: Math.max(-1, Math.min(1, analysis.highlights * 0.3)),
      shadows: Math.max(-1, Math.min(1, analysis.shadows * 0.3)),
    }));

    saveState();
    displayToast("Auto Tone Applied");
  };

  const analyzeImageColor = (): {
    temp: number;
    tint: number;
    saturation: number;
  } => {
    const gl = glRef.current;
    const canvas = canvasRef.current;

    if (!gl || !canvas) return { temp: 0, tint: 0, saturation: 0 };

    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    let rSum = 0,
      gSum = 0,
      bSum = 0;
    let satSum = 0;
    let count = 0;

    // Sample every 10th pixel for performance
    for (let i = 0; i < pixels.length; i += 40) {
      const r = pixels[i] / 255;
      const g = pixels[i + 1] / 255;
      const b = pixels[i + 2] / 255;

      // Skip very dark or very bright pixels
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      if (luma < 0.1 || luma > 0.9) continue;

      rSum += r;
      gSum += g;
      bSum += b;

      // Calculate saturation
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const sat = max > 0 ? (max - min) / max : 0;
      satSum += sat;

      count++;
    }

    if (count === 0) return { temp: 0, tint: 0, saturation: 0 };

    const avgR = rSum / count;
    const avgG = gSum / count;
    const avgB = bSum / count;
    const avgSat = satSum / count;

    // Calculate color temperature adjustment
    const temp = (avgB - avgR) * 0.3;

    // Calculate tint adjustment (green-magenta)
    const tint = (avgG - (avgR + avgB) / 2) * 0.2;

    // Calculate saturation adjustment
    const targetSat = 0.4;
    const saturation = avgSat < targetSat ? (targetSat - avgSat) * 0.5 : 0;

    return { temp, tint, saturation };
  };

  const handleAutoColor = () => {
    render(); // Ensure we have current pixels
    const analysis = analyzeImageColor();

    setState((prev) => ({
      ...prev,
      temp: Math.max(-0.5, Math.min(0.5, analysis.temp)),
      tint: Math.max(-0.5, Math.min(0.5, analysis.tint)),
      saturation: Math.max(-1, Math.min(1, analysis.saturation)),
    }));

    saveState();
    displayToast("Auto Color Applied");
  };

  const handleSliderChange = (key: keyof State, value: number) => {
    setState((prev) => ({ ...prev, [key]: value }));
  };

  const handleHSLChange = (channel: "h" | "s" | "l", value: number) => {
    setState((prev) => ({
      ...prev,
      hsl: {
        ...prev.hsl,
        [channel]: prev.hsl[channel].map((v, i) =>
          i === prev.activeHSLChannel ? value : v,
        ),
      },
    }));
  };

  const handleColorDotClick = (channel: number) => {
    setState((prev) => ({ ...prev, activeHSLChannel: channel }));
  };

  const handleAspectRatioChange = (ratio: number) => {
    setState((prev) => ({ ...prev, cropAspect: ratio }));
    saveState();
  };

  const handleGridTypeChange = (gridType: string) => {
    setState((prev) => ({ ...prev, gridType }));
    setShowGridMenu(false);
  };

  const handleSave = () => {
    render();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement("a");
    link.download = "AutoHDR_Pro.jpg";
    link.href = canvas.toDataURL("image/jpeg", 0.95);
    link.click();
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      if (url) loadImage(url);
    };
    reader.readAsDataURL(file);
  };

  const handleStraightenStart = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStraightening) return;

    const rect = e.currentTarget.getBoundingClientRect();
    setStraightenStart({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const handleStraightenMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStraightening || !straightenStart) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;

    const rect = e.currentTarget.getBoundingClientRect();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.moveTo(straightenStart.x, straightenStart.y);
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
  };

  const handleStraightenEnd = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStraightening || !straightenStart) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const dx = e.clientX - rect.left - straightenStart.x;
    const dy = e.clientY - rect.top - straightenStart.y;

    const angle = -Math.atan2(dy, dx) * (180 / Math.PI);
    setState((prev) => ({ ...prev, rotate: angle }));
    saveState();

    setStraightenStart(null);
    setIsStraightening(false);

    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    displayToast("Horizon Fixed");
  };

  // Initialize WebGL on mount
  useEffect(() => {
    initWebGL();
    loadImage(DEFAULT_IMG);
    saveState();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Start animation loop
  useEffect(() => {
    if (glRef.current && programRef.current && textureRef.current) {
      animate();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [animate]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => resizeViewport();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [state.cropAspect]);

  const colorDots = [
    "#ff4d4d",
    "#ff9f43",
    "#feca57",
    "#1dd1a1",
    "#00d2d3",
    "#54a0ff",
    "#5f27cd",
    "#ff9ff3",
  ];

  const tabs = [
    { id: "basic", label: "Basic", icon: Sliders },
    { id: "color", label: "Color", icon: Palette },
    { id: "optics", label: "Optics", icon: Eye },
    { id: "geometry", label: "Geometry", icon: CropIcon },
    { id: "format", label: "Format", icon: ImageIcon },
  ];

  const gridOptions = [
    { id: "none", label: "None", icon: EyeOff },
    { header: "Composition" },
    { id: "center", label: "Center", icon: Grid3x3 },
    { id: "thirds", label: "Rule of Thirds", icon: Grid3x3 },
    { id: "golden", label: "Golden Ratio", icon: Compass },
    { id: "fifths", label: "Fifths", icon: Grid3x3 },
    { id: "grid", label: "Grid (8x8)", icon: Grid3x3 },
    { id: "triangle", label: "Golden Triangle", icon: Grid3x3 },
    { id: "diag", label: "Diagonals", icon: Grid3x3 },
    { header: "Proportions" },
    { id: "ar11", label: "1:1 Square", icon: Square },
    { id: "ar43", label: "4:3 Horizontal", icon: ImageIcon },
    { id: "ar34", label: "3:4 Vertical", icon: ImageIcon },
    { id: "ar169", label: "16:9 Landscape", icon: ImageIcon },
    { id: "ar916", label: "9:16 Story", icon: ImageIcon },
  ];

  return (
    <div className="flex h-screen w-screen bg-black text-gray-400 text-[11px]">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Sidebar */}
      <aside className="w-80 bg-[#0f0f0f] border-r border-[#1f1f1f] flex flex-col shrink-0">
        {/* Header */}
        <div className="h-14 flex items-center px-4 border-b border-neutral-900 bg-[#050505]">
          <h1 className="text-white font-bold tracking-widest text-xs">
            AutoHDR Studio{" "}
            <span className="text-neutral-500 font-normal">PRO</span>
          </h1>
          <div className="ml-auto flex gap-1 items-center">
            <button
              onClick={handleUndo}
              disabled={undoStack.length <= 1}
              className="w-7 h-7 flex items-center justify-center rounded hover:bg-[#222] disabled:opacity-30"
            >
              <Undo2 size={14} />
            </button>
            <button
              onClick={handleRedo}
              disabled={redoStack.length === 0}
              className="w-7 h-7 flex items-center justify-center rounded hover:bg-[#222] disabled:opacity-30"
            >
              <Redo2 size={14} />
            </button>
            <div className="w-px h-4 bg-neutral-800 mx-1" />
            <button
              onClick={handleReset}
              className="w-7 h-7 flex items-center justify-center rounded hover:bg-[#222]"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>

        {/* Histogram */}
        <div className="relative bg-black border-b border-neutral-900 h-28 w-full">
          <canvas
            ref={histogramCanvasRef}
            className="w-full h-full block"
            width={320}
            height={112}
          />
          <div className="absolute top-2 left-2 text-[9px] text-neutral-600 font-mono">
            RGB PARADE
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-neutral-900 bg-[#050505] text-[9px] uppercase font-bold tracking-wider">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 py-3 text-center flex flex-col items-center gap-1 border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? "text-white border-white"
                    : "text-gray-500 border-transparent hover:text-gray-300"
                }`}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === "basic" && (
            <div>
              <div className="grid grid-cols-2 gap-2 mb-6">
                <button
                  onClick={handleAutoTone}
                  className="bg-neutral-900 hover:bg-neutral-800 text-white text-[10px] uppercase py-2 rounded border border-neutral-800 flex items-center justify-center transition-colors"
                >
                  <Wand2 size={14} className="mr-2 text-yellow-500" />
                  Auto Tone
                </button>
                <button
                  onClick={handleAutoColor}
                  className="bg-neutral-900 hover:bg-neutral-800 text-white text-[10px] uppercase py-2 rounded border border-neutral-800 flex items-center justify-center transition-colors"
                >
                  <Droplet size={14} className="mr-2 text-blue-500" />
                  Auto Color
                </button>
              </div>

              <ControlSlider
                label="EXPOSURE"
                icon={<Sun size={14} />}
                value={state.exposure}
                min={-2}
                max={2}
                step={0.01}
                onChange={(v) => handleSliderChange("exposure", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="CONTRAST"
                icon={<CircleIcon size={14} />}
                value={state.contrast}
                min={0.5}
                max={1.5}
                step={0.01}
                onChange={(v) => handleSliderChange("contrast", v)}
                onChangeEnd={saveState}
              />

              <div className="h-px bg-neutral-900 my-4" />

              <ControlSlider
                label="HIGHLIGHTS"
                icon={<Zap size={14} />}
                value={state.highlights}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("highlights", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="SHADOWS"
                icon={<Cloud size={14} />}
                value={state.shadows}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("shadows", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="WHITES"
                icon={<Square size={14} />}
                value={state.whites}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("whites", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="BLACKS"
                icon={<Square size={14} fill="currentColor" />}
                value={state.blacks}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("blacks", v)}
                onChangeEnd={saveState}
              />
            </div>
          )}

          {activeTab === "color" && (
            <div>
              <ControlSlider
                label="TEMP"
                icon={<Thermometer size={14} />}
                value={state.temp}
                min={-0.5}
                max={0.5}
                step={0.01}
                onChange={(v) => handleSliderChange("temp", v)}
                onChangeEnd={saveState}
                className="slider-temp"
              />

              <ControlSlider
                label="TINT"
                icon={<Droplet size={14} />}
                value={state.tint}
                min={-0.5}
                max={0.5}
                step={0.01}
                onChange={(v) => handleSliderChange("tint", v)}
                onChangeEnd={saveState}
                className="slider-tint"
              />

              <div className="h-px bg-neutral-900 my-4" />

              <ControlSlider
                label="SATURATION"
                icon={<Palette size={14} />}
                value={state.saturation}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("saturation", v)}
                onChangeEnd={saveState}
                className="slider-sat"
              />

              <ControlSlider
                label="VIBRANCE"
                icon={<Palette size={14} />}
                value={state.vibrance}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("vibrance", v)}
                onChangeEnd={saveState}
                className="slider-sat"
              />

              <div className="mt-8 pt-4 border-t border-neutral-900">
                <div className="text-[10px] font-bold text-white tracking-widest mb-4">
                  <Sliders size={14} className="inline mr-2" />
                  COLOR EDITOR
                </div>

                <div className="flex justify-between mb-6 px-1">
                  {colorDots.map((color, i) => (
                    <button
                      key={i}
                      onClick={() => handleColorDotClick(i)}
                      className={`w-6 h-6 rounded-full border-2 transition-transform hover:scale-110 ${
                        state.activeHSLChannel === i
                          ? "border-white scale-110"
                          : "border-transparent"
                      }`}
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>

                <div className="p-3 bg-neutral-900 rounded border border-neutral-800">
                  <ControlSlider
                    label="HUE"
                    value={state.hsl.h[state.activeHSLChannel]}
                    min={-0.5}
                    max={0.5}
                    step={0.01}
                    onChange={(v) => handleHSLChange("h", v)}
                    onChangeEnd={saveState}
                    className="slider-hue"
                  />

                  <ControlSlider
                    label="SATURATION"
                    value={state.hsl.s[state.activeHSLChannel]}
                    min={-1}
                    max={1}
                    step={0.01}
                    onChange={(v) => handleHSLChange("s", v)}
                    onChangeEnd={saveState}
                  />

                  <ControlSlider
                    label="LIGHTNESS"
                    value={state.hsl.l[state.activeHSLChannel]}
                    min={-1}
                    max={1}
                    step={0.01}
                    onChange={(v) => handleHSLChange("l", v)}
                    onChangeEnd={saveState}
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === "optics" && (
            <div>
              <ControlSlider
                label="DISTORTION"
                icon={<Eye size={14} />}
                value={state.distortion}
                min={-0.5}
                max={0.5}
                step={0.01}
                onChange={(v) => handleSliderChange("distortion", v)}
                onChangeEnd={saveState}
              />

              <div className="h-px bg-neutral-900 my-4" />

              <ControlSlider
                label="VIGNETTE"
                icon={<CircleIcon size={14} />}
                value={state.vignette}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("vignette", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="MIDPOINT"
                icon={<Move size={14} />}
                value={state.vignetteMid}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("vignetteMid", v)}
                onChangeEnd={saveState}
              />
            </div>
          )}

          {activeTab === "geometry" && (
            <div>
              <div className="bg-[#0f0f0f] p-3 rounded mb-4 border border-neutral-800">
                <button
                  onClick={() => setIsStraightening(!isStraightening)}
                  className={`w-full flex items-center justify-center bg-neutral-900 hover:bg-neutral-800 text-white text-[10px] uppercase py-3 rounded transition border ${
                    isStraightening ? "border-white" : "border-transparent"
                  }`}
                >
                  <Ruler size={14} className="mr-2" />
                  Draw Horizon
                </button>
                <p className="text-[9px] text-center mt-2 text-neutral-500">
                  Drag a line across the horizon to level
                </p>
              </div>

              <ControlSlider
                label="ROTATE"
                icon={<RotateCw size={14} />}
                value={state.rotate}
                min={-45}
                max={45}
                step={0.1}
                onChange={(v) => handleSliderChange("rotate", v)}
                onChangeEnd={saveState}
                suffix="°"
              />

              <div className="h-px bg-neutral-900 my-4" />

              <ControlSlider
                label="VERTICAL"
                icon={<Move size={14} />}
                value={state.perspectiveY}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("perspectiveY", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="HORIZONTAL"
                icon={<Move size={14} />}
                value={state.perspectiveX}
                min={-1}
                max={1}
                step={0.01}
                onChange={(v) => handleSliderChange("perspectiveX", v)}
                onChangeEnd={saveState}
              />

              <ControlSlider
                label="SCALE"
                icon={<Maximize size={14} />}
                value={state.scale}
                min={0.5}
                max={3}
                step={0.01}
                onChange={(v) => handleSliderChange("scale", v)}
                onChangeEnd={saveState}
              />
            </div>
          )}

          {activeTab === "format" && (
            <div>
              <p className="text-[9px] text-neutral-500 mb-3 text-center uppercase tracking-wider">
                Select Crop Aspect Ratio
              </p>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { ar: 0, label: "Original", icon: ImageIcon },
                  { ar: 1, label: "1:1 Square", icon: Square },
                  { ar: 1.777, label: "16:9", icon: ImageIcon },
                  { ar: 1.333, label: "4:3", icon: ImageIcon },
                  { ar: 0.666, label: "2:3", icon: ImageIcon },
                  { ar: 0.5625, label: "9:16", icon: ImageIcon },
                ].map(({ ar, label, icon: Icon }) => (
                  <button
                    key={ar}
                    onClick={() => handleAspectRatioChange(ar)}
                    className={`bg-neutral-900 hover:bg-neutral-800 text-white py-4 rounded border transition flex flex-col items-center ${
                      state.cropAspect === ar
                        ? "border-white"
                        : "border-transparent"
                    }`}
                  >
                    <Icon size={18} className="mb-2" />
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-neutral-900 bg-[#050505]">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-neutral-800 hover:bg-neutral-700 text-white text-[11px] uppercase font-bold py-3 rounded mb-2"
          >
            Load Image
          </button>
          <button
            onClick={handleSave}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white text-[11px] uppercase font-bold py-3 rounded shadow-lg shadow-blue-900/20"
          >
            <Download size={14} className="inline mr-2" />
            Export Image
          </button>
        </div>
      </aside>

      {/* Main viewport */}
      <main className="flex-1 flex items-center justify-center relative bg-black">
        {/* Grid menu */}
        <div className="absolute top-4 left-4 z-50">
          <div className="relative">
            <button
              onMouseEnter={() => setShowGridMenu(true)}
              onMouseLeave={() => setShowGridMenu(false)}
              className="w-10 h-10 rounded-full bg-black/80 border border-white/20 flex items-center justify-center hover:w-28 transition-all duration-300 overflow-hidden"
            >
              <div className="w-10 h-10 flex items-center justify-center shrink-0">
                <Grid3x3 size={16} className="text-white" />
              </div>
              <span className="text-white text-[10px] font-bold tracking-widest pr-3 opacity-0 hover:opacity-100 transition-opacity">
                GUIDES
              </span>
            </button>

            {showGridMenu && (
              <div
                onMouseEnter={() => setShowGridMenu(true)}
                onMouseLeave={() => setShowGridMenu(false)}
                className="absolute top-12 left-0 w-56 bg-[#0a0a0a]/95 backdrop-blur-md border border-[#333] rounded-lg p-1.5 shadow-2xl"
              >
                {gridOptions.map((option, i) => {
                  if ("header" in option) {
                    return (
                      <div
                        key={i}
                        className="text-[9px] text-gray-500 uppercase tracking-widest px-3 py-2 font-bold border-t border-[#222] mt-1 first:border-t-0 first:mt-0"
                      >
                        {option.header}
                      </div>
                    );
                  }

                  const Icon = option.icon;
                  return (
                    <button
                      key={option.id}
                      onClick={() => handleGridTypeChange(option.id)}
                      className={`w-full flex items-center px-3 py-2 text-[11px] rounded transition-colors ${
                        state.gridType === option.id
                          ? "bg-[#222] text-blue-500 font-semibold"
                          : "text-gray-400 hover:bg-[#222] hover:text-white"
                      }`}
                    >
                      <Icon size={12} className="mr-2" />
                      {option.label}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Canvas wrapper */}
        <div ref={wrapperRef} className="relative shadow-2xl">
          <canvas ref={canvasRef} className="block" />
          <GridOverlay gridType={state.gridType} />
          <canvas
            ref={overlayCanvasRef}
            className="absolute inset-0"
            style={{
              pointerEvents: isStraightening ? "auto" : "none",
              cursor: isStraightening ? "crosshair" : "default",
            }}
            onMouseDown={handleStraightenStart}
            onMouseMove={handleStraightenMove}
            onMouseUp={handleStraightenEnd}
          />
        </div>

        {/* Toast notification */}
        <div
          className={`absolute top-5 right-5 bg-neutral-800 text-white px-4 py-2 rounded text-xs shadow-xl flex items-center border border-neutral-700 transform transition-transform ${
            showToast.show ? "translate-y-0" : "-translate-y-[150%]"
          }`}
        >
          <div className="w-2 h-2 bg-green-500 rounded-full mr-2" />
          {showToast.message}
        </div>

        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black z-50">
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent mb-3" />
              <span className="text-xs tracking-widest text-neutral-500 uppercase">
                Loading Darkroom
              </span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Control Slider Component
interface ControlSliderProps {
  label: string;
  icon?: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  onChangeEnd?: () => void;
  className?: string;
  suffix?: string;
}

function ControlSlider({
  label,
  icon,
  value,
  min,
  max,
  step,
  onChange,
  onChangeEnd,
  className = "",
  suffix = "",
}: ControlSliderProps) {
  return (
    <div className="mb-3.5">
      <div className="flex justify-between items-center mb-1.5 text-[10px] font-semibold tracking-wider text-gray-400">
        <div className="flex items-center gap-2">
          {icon}
          {label}
        </div>
        <span>
          {value.toFixed(2)}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        onMouseUp={onChangeEnd}
        onTouchEnd={onChangeEnd}
        className={`w-full h-[18px] appearance-none bg-transparent cursor-pointer ${className}`}
      />
    </div>
  );
}

// Grid Overlay Component
function GridOverlay({ gridType }: { gridType: string }) {
  if (gridType === "none") return null;

  return (
    <svg className="absolute inset-0 pointer-events-none w-full h-full z-50">
      {gridType === "center" && (
        <g
          stroke="rgba(255,255,255,0.7)"
          strokeWidth="1.5"
          fill="none"
          opacity="0.9"
        >
          <line x1="50%" y1="0" x2="50%" y2="100%" />
          <line x1="0" y1="50%" x2="100%" y2="50%" />
        </g>
      )}

      {gridType === "thirds" && (
        <g
          stroke="rgba(255,255,255,0.7)"
          strokeWidth="1.5"
          fill="none"
          opacity="0.9"
        >
          <line x1="33.33%" y1="0" x2="33.33%" y2="100%" />
          <line x1="66.66%" y1="0" x2="66.66%" y2="100%" />
          <line x1="0" y1="33.33%" x2="100%" y2="33.33%" />
          <line x1="0" y1="66.66%" x2="100%" y2="66.66%" />
        </g>
      )}

      {gridType === "golden" && (
        <g stroke="rgba(255,215,0,0.8)" strokeWidth="1.5" fill="none">
          <line x1="38.2%" y1="0" x2="38.2%" y2="100%" />
          <line x1="61.8%" y1="0" x2="61.8%" y2="100%" />
          <line x1="0" y1="38.2%" x2="100%" y2="38.2%" />
          <line x1="0" y1="61.8%" x2="100%" y2="61.8%" />
        </g>
      )}

      {/* Add more grid types as needed */}
    </svg>
  );
}
