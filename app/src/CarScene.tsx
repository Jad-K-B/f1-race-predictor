import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { RotateCcw, Move3D, Sun, Camera, Check } from "lucide-react";
gsap.registerPlugin(ScrollTrigger);

type Rig = {
  view: (index: number) => void;
  light: (bright: boolean) => void;
  interact: () => void;
};
export default function CarScene({ motion }: { motion: boolean }) {
  const host = useRef<HTMLDivElement>(null),
    rig = useRef<Rig | null>(null);
  const [state, setState] = useState("Loading concept"),
    [active, setActive] = useState(false),
    [bright, setBright] = useState(false),
    [view, setView] = useState(0);
  useEffect(() => {
    setActive(false);
    setBright(false);
    setView(0);
    const element = host.current!;
    delete element.dataset.loaded;
    setState("Loading concept");
    let dirty = true,
      disposed = false,
      visible = true,
      frame = 0,
      frames = 0,
      interacting = false,
      loaded = false;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      setState("3D unavailable");
      return;
    }
    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, window.innerWidth < 700 ? 1.25 : 1.7),
    );
    renderer.setClearColor(0x15191b, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.88;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 2026 racing concept by Qvist_designs",
    );
    renderer.domElement.setAttribute("role", "img");
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene(),
      camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100),
      car = new THREE.Group();
    scene.add(car);
    const environment = new RoomEnvironment(),
      pmrem = new THREE.PMREMGenerator(renderer),
      environmentMap = pmrem.fromScene(environment, 0.04);
    scene.environment = environmentMap.texture;
    environment.dispose();
    const key = new THREE.DirectionalLight(0xf3f7fa, 3);
    key.position.set(-3, 8, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, {
      left: -6,
      right: 6,
      top: 6,
      bottom: -6,
      near: 0.1,
      far: 30,
    });
    key.shadow.bias = -0.0004;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xd9fb4d, 1.5);
    rim.position.set(4, 3, -4);
    scene.add(rim);
    const fill = new THREE.DirectionalLight(0x9acbff, 1);
    fill.position.set(-5, 2, -2);
    scene.add(fill);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.ShadowMaterial({ opacity: 0.48 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener("change", () => {
      dirty = true;
    });
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.enableZoom = false;
    controls.enabled = false;
    controls.minPolarAngle = 0.2;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.set(-0.7, 0.15, 0);
    const pose = { turn: 0, lift: 0, dolly: 0 };
    const lighting = { blend: 0 };
    let cameraTween: gsap.core.Tween | undefined;
    const applyView = (index: number, animate = true) => {
      const mobile = element.clientWidth < 700,
        ratio = element.clientWidth / element.clientHeight;
      const scale = Math.max(1, 1 / ratio),
        positions = [
          [-6, 2.9, 6],
          [0, 8, 0.5],
          [-0.3, 1.7, 8],
        ];
      const p = positions[index];
      controls.target.set(mobile ? 0 : -1, mobile ? 0.5 : 0.8, 0);
      cameraTween?.kill();
      const target = { x: p[0] * scale, y: p[1] * scale, z: p[2] * scale };
      if (animate && motion)
        cameraTween = gsap.to(camera.position, {
          ...target,
          duration: 1.2,
          ease: "power3.inOut",
          onUpdate: () => {
            controls.update();
          },
        });
      else {
        camera.position.set(target.x, target.y, target.z);
        controls.update();
      }
    };
    let currentView = 0;
    const resize = () => {
      dirty = true;
      const w = element.clientWidth,
        h = element.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      applyView(currentView, false);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    const scroll = motion
      ? gsap.to(pose, {
          turn: 1.1,
          lift: 0.18,
          dolly: 0.18,
          ease: "none",
          scrollTrigger: {
            trigger: "#cinema",
            start: "top top",
            end: "bottom top",
            scrub: 1,
          },
        })
      : null;
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const silver = new THREE.MeshPhysicalMaterial({
      color: 0x859298,
      metalness: 0.9,
      roughness: 0.3,
      clearcoat: 1,
      clearcoatRoughness: 0.23,
    });
    const acid = new THREE.MeshPhysicalMaterial({
      color: 0xd6f63c,
      metalness: 0.22,
      roughness: 0.32,
      clearcoat: 0.8,
    });
    const carbon = new THREE.MeshStandardMaterial({
      color: 0x242a2e,
      metalness: 0.55,
      roughness: 0.4,
    });
    const rubber = new THREE.MeshStandardMaterial({
      color: 0x111417,
      metalness: 0.08,
      roughness: 0.73,
    });
    const metal = new THREE.MeshStandardMaterial({
      color: 0x919caa,
      metalness: 0.88,
      roughness: 0.22,
    });
    loader.load(
      "/assets/formation-car.glb",
      (gltf) => {
        if (disposed) {
          gltf.scene.traverse((o) => {
            if (o instanceof THREE.Mesh) {
              o.geometry.dispose();
              const list = Array.isArray(o.material)
                ? o.material
                : [o.material];
              list.forEach((m) => m.dispose());
            }
          });
          return;
        }
        gltf.scene.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            const old = Array.isArray(object.material)
              ? object.material
              : [object.material];
            old.forEach((m) => m.dispose());
            const name = object.name.toLowerCase();
            object.material = name.includes("wheel")
              ? rubber
              : name.includes("suspension") || name.includes("exhaust")
                ? metal
                : name.includes("wing") || name.includes("floor")
                  ? carbon
                  : name.includes("halo") || name.includes("mirror")
                    ? acid
                    : silver;
            object.castShadow = true;
            object.receiveShadow = true;
          }
        });
        const box = new THREE.Box3().setFromObject(gltf.scene),
          size = box.getSize(new THREE.Vector3()),
          center = box.getCenter(new THREE.Vector3());
        const scale = 6 / Math.max(size.x, size.z);
        gltf.scene.scale.setScalar(scale);
        gltf.scene.position.set(
          -center.x * scale,
          -box.min.y * scale,
          -center.z * scale,
        );
        car.add(gltf.scene);
        loaded = true;
        dirty = true;
        setState("Concept ready");
        element.dataset.loaded = "true";
        element.dataset.triangles = String(renderer.info.render.triangles);
      },
      undefined,
      () => {
        if (!disposed) setState("3D unavailable");
      },
    );
    rig.current = {
      view: (index) => {
        currentView = index;
        applyView(index);
      },
      light: (bright) => {
        // Reflections dominate the metallic body, so lift the environment too.
        gsap.to(lighting, {
          blend: bright ? 1 : 0,
          duration: motion ? 0.7 : 0,
          ease: "power2.inOut",
          overwrite: true,
          onUpdate: () => {
            renderer.toneMappingExposure = 0.88 + lighting.blend * 0.26;
            scene.environmentIntensity = 1 + lighting.blend * 0.65;
            key.intensity = 3 + lighting.blend * 0.7;
            fill.intensity = 1 + lighting.blend;
            dirty = true;
          },
        });
      },
      interact: () => {
        interacting = !interacting;
        controls.enabled = interacting;
        renderer.domElement.style.touchAction = interacting ? "none" : "pan-y";
      },
    };
    const visibility = new IntersectionObserver(
      (entries) => {
        visible = entries[0].isIntersecting;
      },
      { rootMargin: "80px" },
    );
    visibility.observe(element);
    const onLost = (event: Event) => {
      event.preventDefault();
      setState("3D unavailable");
    };
    renderer.domElement.addEventListener("webglcontextlost", onLost);
    let last = 0;
    const draw = (time: number) => {
      frame = requestAnimationFrame(draw);
      if (!visible || document.hidden || time - last < 30) return;
      last = time;
      controls.update();
      if (!motion && !dirty && !interacting) return;
      if (!interacting) {
        car.rotation.y =
          pose.turn + (motion ? Math.sin(time * 0.00013) * 0.025 : 0);
        car.position.y = pose.lift;
        camera.zoom = 1 + pose.dolly;
        camera.updateProjectionMatrix();
      }
      renderer.render(scene, camera);
      dirty = false;
      frames++;
      element.dataset.frames = String(frames);
      if (loaded)
        element.dataset.triangles = String(renderer.info.render.triangles);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      cameraTween?.kill();
      scroll?.scrollTrigger?.kill();
      scroll?.kill();
      gsap.killTweensOf(lighting);
      observer.disconnect();
      visibility.disconnect();
      controls.dispose();
      rig.current = null;
      scene.traverse((o) => {
        if (o instanceof THREE.Mesh) o.geometry.dispose();
      });
      [silver, acid, carbon, rubber, metal, ground.material].forEach((m) =>
        m.dispose(),
      );
      renderer.domElement.removeEventListener("webglcontextlost", onLost);
      environmentMap.dispose();
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [motion]);
  return (
    <div
      className={`car-experience ${state === "Concept ready" ? "loaded" : ""}`}
    >
      <picture>
        <source
          media="(max-width: 700px)"
          srcSet="/assets/car-fallback-mobile.png"
        />
        <img
          className="car-poster"
          src="/assets/car-fallback.png"
          alt="Silver and chartreuse open-wheel racing concept"
        />
      </picture>
      <div
        ref={host}
        className={`car-canvas ${active ? "interactive" : ""}`}
        data-testid="car-canvas"
      />
      {state !== "Concept ready" && (
        <div className="car-status mono">
          <span className="status-dot muted" />
          {state}
        </div>
      )}
      <div className="scene-controls" aria-label="3D car controls">
        <button
          title={active ? "Stop orbit interaction" : "Orbit car"}
          aria-label="Orbit car"
          disabled={state !== "Concept ready"}
          aria-pressed={active}
          onClick={() => {
            rig.current?.interact();
            setActive(!active);
          }}
        >
          {active ? <Check size={18} /> : <Move3D size={18} />}
        </button>
        <button
          title="Change camera angle"
          aria-label="Change camera angle"
          disabled={state !== "Concept ready"}
          onClick={() => {
            const next = (view + 1) % 3;
            setView(next);
            rig.current?.view(next);
          }}
        >
          <Camera size={18} />
        </button>
        <button
          title="Lighting"
          aria-label="Lighting"
          disabled={state !== "Concept ready"}
          aria-pressed={bright}
          onClick={() => {
            rig.current?.light(!bright);
            setBright(!bright);
          }}
        >
          <Sun size={18} />
        </button>
        <button
          title="Reset camera"
          aria-label="Reset camera"
          disabled={state !== "Concept ready"}
          onClick={() => {
            setView(0);
            rig.current?.view(0);
          }}
        >
          <RotateCcw size={18} />
        </button>
      </div>
    </div>
  );
}
