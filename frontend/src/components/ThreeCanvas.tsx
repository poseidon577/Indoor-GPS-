import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SimulationState } from "../types";
import { REUSE_COLOURS_HEX, reuseColours } from "../lightLayout";

export interface PositionMarker {
  estimated: [number, number, number];
  truth: [number, number, number];
}

interface ThreeCanvasProps {
  state: SimulationState;
  marker?: PositionMarker | null;
}

export default function ThreeCanvas({ state, marker }: ThreeCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  const rxMeshRef = useRef<THREE.Group | null>(null);
  const rxFovConeRef = useRef<THREE.Mesh | null>(null);
  const ledsGroupRef = useRef<THREE.Group | null>(null);
  const obstaclesGroupRef = useRef<THREE.Group | null>(null);
  const raysGroupRef = useRef<THREE.Group | null>(null);
  const trajectoryLineRef = useRef<THREE.Line | null>(null);
  const roomGroupRef = useRef<THREE.Group | null>(null);
  const locEstGroupRef = useRef<THREE.Group | null>(null);
  const locErrorLineRef = useRef<THREE.Line | null>(null);

  // ─── 1. Initial Scene Setup (runs once) ─────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080d1a);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
    camera.position.set(10, 9, 10);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 3;
    controls.maxDistance = 30;
    controls.target.set(2.5, 1.5, 2.5);
    controls.update();
    controlsRef.current = controls;

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.3));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(8, 12, 8);
    dir.castShadow = true;
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0x3a8bff, 0.25);
    fill.position.set(-5, 4, -5);
    scene.add(fill);

    // Floor plane
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(50, 50),
      new THREE.MeshStandardMaterial({ color: 0x0e1929, roughness: 0.9, metalness: 0.0 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Grid
    const gridHelper = new THREE.GridHelper(50, 100, 0x1a2744, 0x111a2e);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    // Axes
    const axesHelper = new THREE.AxesHelper(0.8);
    axesHelper.position.set(0.1, 0.01, 0.1);
    scene.add(axesHelper);

    // Groups
    const roomGroup = new THREE.Group();
    scene.add(roomGroup);
    roomGroupRef.current = roomGroup;

    const ledsGroup = new THREE.Group();
    scene.add(ledsGroup);
    ledsGroupRef.current = ledsGroup;

    const obstaclesGroup = new THREE.Group();
    scene.add(obstaclesGroup);
    obstaclesGroupRef.current = obstaclesGroup;

    const raysGroup = new THREE.Group();
    scene.add(raysGroup);
    raysGroupRef.current = raysGroup;

    // ── Receiver model ──────────────────────────────────────────────────
    const rxGroup = new THREE.Group();

    // Metallic base disc
    const baseMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.16, 0.07, 20),
      new THREE.MeshStandardMaterial({ color: 0x1d6ae5, metalness: 0.85, roughness: 0.15 })
    );
    baseMesh.position.y = 0.035;
    rxGroup.add(baseMesh);

    // APD sensor surface glow
    const pdMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.075, 0.075, 0.015, 20),
      new THREE.MeshBasicMaterial({ color: 0x58d6ff })
    );
    pdMesh.position.y = 0.08;
    rxGroup.add(pdMesh);

    // Orientation pointer (cone pointing up)
    const arrowMesh = new THREE.Mesh(
      new THREE.ConeGeometry(0.04, 0.18, 10),
      new THREE.MeshBasicMaterial({ color: 0xffd966 })
    );
    arrowMesh.position.y = 0.26;
    rxGroup.add(arrowMesh);

    scene.add(rxGroup);
    rxMeshRef.current = rxGroup;

    // FOV cone (opens upward from receiver top)
    const fovRad = 1.0 * Math.tan(((state.receiver.fov / 2) * Math.PI) / 180);
    const fovGeo = new THREE.ConeGeometry(fovRad, 1.0, 32, 1, true);
    fovGeo.translate(0, 0.5, 0);
    const fovCone = new THREE.Mesh(
      fovGeo,
      new THREE.MeshBasicMaterial({
        color: 0x58d6ff, transparent: true, opacity: 0.1,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    rxGroup.add(fovCone);
    rxFovConeRef.current = fovCone;

    // Trajectory line
    const trajLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x2ea44f, linewidth: 2 })
    );
    scene.add(trajLine);
    trajectoryLineRef.current = trajLine;

    // ── Estimated-position marker (Module 4 localization) ───────────────
    const locGroup = new THREE.Group();
    locGroup.visible = false;

    // Outer glow ring
    const ringGeo = new THREE.TorusGeometry(0.22, 0.012, 12, 48);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xff7c20, transparent: true, opacity: 0.85 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    locGroup.add(ring);

    // Inner pulsing core sphere
    const coreMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xff9944 })
    );
    locGroup.add(coreMesh);

    // Second outer ring (cross-axis)
    const ring2 = new THREE.Mesh(
      new THREE.TorusGeometry(0.22, 0.007, 8, 48),
      new THREE.MeshBasicMaterial({ color: 0xff9944, transparent: true, opacity: 0.45 })
    );
    ring2.rotation.z = Math.PI / 2;
    locGroup.add(ring2);

    scene.add(locGroup);
    locEstGroupRef.current = locGroup;

    // Error line from estimated to true position
    const errLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineDashedMaterial({ color: 0xff7c20, dashSize: 0.06, gapSize: 0.04, opacity: 0.6, transparent: true })
    );
    scene.add(errLine);
    locErrorLineRef.current = errLine;

    // Resize handler
    const onResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // Render loop — includes estimated-position pulse animation
    let animId: number;
    let tick = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      tick += 0.04;
      controls.update();
      // Pulse the localization marker core
      if (locGroup.visible && locGroup.children[1]) {
        const s = 1.0 + 0.25 * Math.sin(tick * 3.0);
        locGroup.children[1].scale.setScalar(s);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── 2. Room Wireframe ───────────────────────────────────────────────────
  useEffect(() => {
    const group = roomGroupRef.current;
    if (!group) return;

    while (group.children.length > 0) group.remove(group.children[0]);

    const { width, length, height } = state.room;

    // Bright blue room edge wireframe — defines the room shape clearly
    const w = width, l = length, h = height;
    const pts: number[] = [
      // Bottom face
      0,0,0, w,0,0,  w,0,0, w,0,l,  w,0,l, 0,0,l,  0,0,l, 0,0,0,
      // Top face
      0,h,0, w,h,0,  w,h,0, w,h,l,  w,h,l, 0,h,l,  0,h,l, 0,h,0,
      // Vertical edges
      0,0,0, 0,h,0,  w,0,0, w,h,0,  w,0,l, w,h,l,  0,0,l, 0,h,l,
    ];
    const wallGeo = new THREE.BufferGeometry();
    wallGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
    const roomEdges = new THREE.LineSegments(wallGeo, new THREE.LineBasicMaterial({ color: 0x2563eb }));
    group.add(roomEdges);

    // Corner pillars
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x1d3461, metalness: 0.4, roughness: 0.6 });
    [[0, 0], [w, 0], [0, l], [w, l]].forEach(([px, pz]) => {
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, h, 8), pillarMat);
      pillar.position.set(px, h / 2, pz);
      group.add(pillar);
    });

    // Ceiling panel (semi-transparent)
    const ceil = new THREE.Mesh(
      new THREE.PlaneGeometry(w, l),
      new THREE.MeshStandardMaterial({ color: 0x1a2744, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false })
    );
    ceil.rotation.x = Math.PI / 2;
    ceil.position.set(w / 2, h, l / 2);
    group.add(ceil);

    // Translucent wall panels
    const wallAlpha = 0.045;
    const wallColor = 0x1e3a6e;
    const wallConfigs: { w: number; h: number; pos: [number, number, number]; ry: number }[] = [
      { w, h, pos: [w / 2, h / 2, 0],   ry: 0 },
      { w, h, pos: [w / 2, h / 2, l],   ry: Math.PI },
      { w: l, h, pos: [0,   h / 2, l / 2], ry: Math.PI / 2 },
      { w: l, h, pos: [w,   h / 2, l / 2], ry: -Math.PI / 2 },
    ];
    wallConfigs.forEach(({ w: pw, h: ph, pos, ry }) => {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(pw, ph),
        new THREE.MeshBasicMaterial({ color: wallColor, transparent: true, opacity: wallAlpha, side: THREE.DoubleSide, depthWrite: false })
      );
      mesh.position.set(...pos);
      mesh.rotation.y = ry;
      group.add(mesh);
    });

    // Update camera look-at
    if (controlsRef.current) controlsRef.current.target.set(w / 2, h / 3, l / 2);
    if (cameraRef.current)   cameraRef.current.position.set(w * 2.0, h * 2.2, l * 2.0);
  }, [state.room]);

  // ─── 3. Obstacles ────────────────────────────────────────────────────────
  useEffect(() => {
    const group = obstaclesGroupRef.current;
    if (!group) return;
    while (group.children.length > 0) group.remove(group.children[0]);

    state.obstacles.forEach((obs) => {
      const mat = new THREE.MeshStandardMaterial({
        color: 0xc0392b, transparent: true, opacity: 0.7,
        roughness: 0.5, metalness: 0.1
      });

      let geo: THREE.BufferGeometry;
      if (obs.type === "sphere") {
        geo = new THREE.SphereGeometry(obs.scale[0], 24, 24);
      } else if (obs.type === "cylinder") {
        geo = new THREE.CylinderGeometry(obs.scale[0], obs.scale[0], obs.scale[2], 24);
      } else {
        // Python scale [dx, dy, dz], Z is up -> Three.js Y up
        geo = new THREE.BoxGeometry(obs.scale[0], obs.scale[2], obs.scale[1]);
      }

      const mesh = new THREE.Mesh(geo, mat);
      // Python pos [x, y, z_height] => Three.js [x, z_height, y]
      mesh.position.set(obs.position[0], obs.position[2], obs.position[1]);
      mesh.rotation.set(
        THREE.MathUtils.degToRad(obs.rotation[0]),
        THREE.MathUtils.degToRad(obs.rotation[2]),
        THREE.MathUtils.degToRad(obs.rotation[1])
      );
      mesh.castShadow = true;
      group.add(mesh);

      // Wireframe overlay
      const edges = new THREE.EdgesGeometry(geo);
      const wire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xff4444 }));
      wire.position.copy(mesh.position);
      wire.rotation.copy(mesh.rotation);
      group.add(wire);
    });
  }, [state.obstacles]);

  // ─── 4. LEDs + Emission Cones ────────────────────────────────────────────
  useEffect(() => {
    const group = ledsGroupRef.current;
    if (!group) return;
    while (group.children.length > 0) group.remove(group.children[0]);

    // Colour = tone set (four-colour reuse): LEDs with the same colour share frequency tones
    const colourOf = reuseColours(state.leds);
    const many = state.leds.length > 9;  // keep the scene light with big grids

    state.leds.forEach((led) => {
      // Python [x, y, z_height] => Three.js [x, z_height, y]
      const ledGroup = new THREE.Group();
      ledGroup.position.set(led.position[0], led.position[2], led.position[1]);

      const bulbColor = REUSE_COLOURS_HEX[(colourOf[led.id] ?? 1) - 1];

      // Housing disc on ceiling
      const fixture = new THREE.Mesh(
        new THREE.CylinderGeometry(0.14, 0.14, 0.05, 20),
        new THREE.MeshStandardMaterial({ color: 0x374151, metalness: 0.7, roughness: 0.3 })
      );
      ledGroup.add(fixture);

      // Emitter glow bulb (just below disc)
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.065, 14, 14),
        new THREE.MeshBasicMaterial({ color: bulbColor })
      );
      bulb.position.y = -0.05;
      ledGroup.add(bulb);

      // Point light (skipped for big grids — dozens of point lights slow the renderer)
      if (!many) {
        const pt = new THREE.PointLight(bulbColor, 0.45, 4.5);
        pt.position.y = -0.05;
        ledGroup.add(pt);
      }

      // Downward emission cone (apex at LED, base at floor)
      const coneHeight = led.position[2]; // height above floor (Three.js Y)
      if (coneHeight > 0.1) {
        const halfAngleRad = (led.fov / 2 * Math.PI) / 180;
        const coneRadius = coneHeight * Math.tan(halfAngleRad);

        // ConeGeometry default: axis along +Y, apex at +Y/2, base at -Y/2
        // We want apex at top (y=0) and base pointing down (y=-coneHeight)
        const coneGeo = new THREE.ConeGeometry(coneRadius, coneHeight, 32, 1, true);
        // After this translate, apex is at (0,0) and base center is at (0,-coneHeight)
        coneGeo.translate(0, -coneHeight / 2, 0);

        const cone = new THREE.Mesh(
          coneGeo,
          new THREE.MeshBasicMaterial({
            color: bulbColor, transparent: true, opacity: 0.07,
            side: THREE.DoubleSide, depthWrite: false
          })
        );
        if (!many) ledGroup.add(cone);

        // Floor footprint ring
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(coneRadius - 0.02, coneRadius + 0.02, 48),
          new THREE.MeshBasicMaterial({ color: bulbColor, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = -coneHeight + 0.005;
        ledGroup.add(ring);
      }

      group.add(ledGroup);
    });
  }, [state.leds]);

  // ─── 5. Dynamic Frame Updates (Receiver + Rays + Trajectory) ─────────────
  useEffect(() => {
    // 5.1 Receiver
    const rxGroup = rxMeshRef.current;
    if (rxGroup) {
      // Python pos [x, y, z] where z=height => Three.js [x, z_height, y]
      rxGroup.position.set(
        state.receiver.position[0],
        state.receiver.position[2],
        state.receiver.position[1]
      );
      rxGroup.rotation.set(
        THREE.MathUtils.degToRad(state.receiver.roll),
        THREE.MathUtils.degToRad(state.receiver.yaw),
        THREE.MathUtils.degToRad(state.receiver.pitch)
      );

      const fovCone = rxFovConeRef.current;
      if (fovCone) {
        const r = 1.2 * Math.tan(((state.receiver.fov / 2) * Math.PI) / 180);
        fovCone.scale.set(r, 1.2, r);
      }
    }

    // 5.2 Trajectory
    const trajLine = trajectoryLineRef.current;
    if (trajLine && state.trajectoryPoints.length > 1) {
      const pts = state.trajectoryPoints.map(
        (pt) => new THREE.Vector3(pt[0], pt[2], pt[1])
      );
      trajLine.geometry.setFromPoints(pts);
      trajLine.geometry.computeBoundingSphere();
    }

    // 5.3 Optical Rays
    const raysGroup = raysGroupRef.current;
    if (raysGroup) {
      while (raysGroup.children.length > 0) raysGroup.remove(raysGroup.children[0]);

      const rxPos = new THREE.Vector3(
        state.receiver.position[0],
        state.receiver.position[2],
        state.receiver.position[1]
      );

      state.leds.forEach((led) => {
        const isLos = state.losMatrix[led.id];
        const inFov = state.visibilityMatrix[led.id];
        const ledPos = new THREE.Vector3(led.position[0], led.position[2], led.position[1]);

        let rayColor: number;
        if (!isLos) rayColor = 0xe53e3e;
        else if (inFov) rayColor = 0x22c55e;
        else rayColor = 0x64748b;

        const rayGeo = new THREE.BufferGeometry().setFromPoints([ledPos, rxPos]);

        let ray: THREE.Line;
        if (!isLos) {
          const mat = new THREE.LineDashedMaterial({ color: rayColor, dashSize: 0.12, gapSize: 0.08 });
          ray = new THREE.Line(rayGeo, mat);
          ray.computeLineDistances();
        } else {
          ray = new THREE.Line(rayGeo, new THREE.LineBasicMaterial({ color: rayColor }));
        }
        raysGroup.add(ray);
      });
    }
  }, [state.receiver, state.trajectoryPoints, state.leds, state.losMatrix, state.visibilityMatrix, state.blockingObstacles, state.obstacles]);

  // ─── 6. Localization Estimated Position Marker ───────────────────────────
  useEffect(() => {
    const locGroup = locEstGroupRef.current;
    const errLine = locErrorLineRef.current;

    if (!locGroup) return;

    if (!marker) {
      locGroup.visible = false;
      if (errLine) errLine.visible = false;
      return;
    }

    // Python [x, y, z_height] → Three.js [x, z_height, y]
    const ep = marker.estimated;
    const tp = marker.truth;

    locGroup.position.set(ep[0], ep[2], ep[1]);
    locGroup.visible = true;

    // Error line between estimated and true position
    if (errLine) {
      const estVec = new THREE.Vector3(ep[0], ep[2], ep[1]);
      const trueVec = new THREE.Vector3(tp[0], tp[2], tp[1]);
      errLine.geometry.setFromPoints([estVec, trueVec]);
      errLine.computeLineDistances();
      errLine.visible = true;
    }
  }, [marker?.estimated[0], marker?.estimated[1], marker?.estimated[2], marker?.truth[0], marker?.truth[1], !!marker]);

  return (
    <div
      id="three-canvas-container"
      className="relative w-full h-full rounded-xl overflow-hidden"
      style={{ background: "radial-gradient(ellipse at 50% 80%, #0d1b3e 0%, #080d1a 100%)" }}
    >
      <div ref={containerRef} className="w-full h-full" />
      <div className="absolute top-3 left-3 text-[10px] font-mono text-blue-400/60 select-none pointer-events-none">
        VLCL · 3D Digital Twin
      </div>
      <div className="absolute top-3 right-3 text-[10px] font-mono text-slate-500 select-none pointer-events-none">
        Drag to orbit · Scroll to zoom
      </div>
      <div className="absolute bottom-3 right-3 flex items-center gap-1.5 text-[10px] font-mono text-emerald-400 select-none pointer-events-none">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        LIVE
      </div>
    </div>
  );
}
