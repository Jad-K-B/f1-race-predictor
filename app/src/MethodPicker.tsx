import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";

type Option = { value: string; label: string; detail: string };

export default function MethodPicker({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
}) {
  const id = useId();
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const selected = options.findIndex((option) => option.value === value);
  const [active, setActive] = useState(selected);
  const typed = useRef({ text: "", time: 0 });

  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  useEffect(() => {
    if (open)
      document
        .getElementById(`${id}-${active}`)
        ?.scrollIntoView({ block: "nearest" });
  }, [open, active, id]);

  function choose(index: number) {
    onChange(options[index].value);
    setOpen(false);
    trigger.current?.focus({ preventScroll: true });
  }

  return (
    <div
      className="method-picker"
      ref={host}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        className="method-trigger"
        role="combobox"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-options`}
        aria-activedescendant={open ? `${id}-${active}` : undefined}
        onClick={() => {
          setActive(selected);
          setOpen(!open);
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            setOpen(false);
            return;
          }
          if (event.key === "Tab") {
            setOpen(false);
            return;
          }
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (open) choose(active);
            else {
              setActive(selected);
              setOpen(true);
            }
            return;
          }
          const move =
            event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
          if (move || event.key === "Home" || event.key === "End") {
            event.preventDefault();
            setActive(
              event.key === "Home"
                ? 0
                : event.key === "End"
                  ? options.length - 1
                  : open
                    ? (active + move + options.length) % options.length
                    : selected,
            );
            setOpen(true);
          } else if (
            event.key.length === 1 &&
            !event.ctrlKey &&
            !event.metaKey &&
            !event.altKey
          ) {
            event.preventDefault();
            const now = Date.now();
            const text =
              (now - typed.current.time < 700 ? typed.current.text : "") +
              event.key.toLowerCase();
            typed.current = { text, time: now };
            const match = options.findIndex((option) =>
              option.label.toLowerCase().startsWith(text),
            );
            if (match >= 0) {
              setActive(match);
              setOpen(true);
            }
          }
        }}
      >
        <span>
          <strong>{options[selected].label}</strong>
          <small>{options[selected].detail}</small>
        </span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && (
        <div
          id={`${id}-options`}
          className="method-options"
          role="listbox"
          aria-label={label}
        >
          {options.map((option, index) => (
            <div
              key={option.value}
              id={`${id}-${index}`}
              role="option"
              aria-selected={value === option.value}
              data-active={active === index}
              onPointerMove={() => setActive(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(index)}
            >
              <span>
                <strong>{option.label}</strong>
                <small>{option.detail}</small>
              </span>
              {value === option.value && <Check size={16} aria-hidden="true" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
