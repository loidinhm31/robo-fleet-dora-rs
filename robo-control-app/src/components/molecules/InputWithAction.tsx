import React, { useState, KeyboardEvent } from "react";
import { LucideIcon, Send } from "lucide-react";

export interface InputWithActionProps {
  value?: string;
  onChange?: (value: string) => void;
  onSubmit: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  icon?: LucideIcon;
  buttonText?: string;
  clearOnSubmit?: boolean;
  className?: string;
}

export const InputWithAction: React.FC<InputWithActionProps> = ({
  value: controlledValue,
  onChange: controlledOnChange,
  onSubmit,
  placeholder = "Enter text...",
  disabled = false,
  icon: Icon = Send,
  buttonText,
  clearOnSubmit = true,
  className = "",
}) => {
  const [internalValue, setInternalValue] = useState("");

  const isControlled = controlledValue !== undefined;
  const value = isControlled ? controlledValue : internalValue;
  const setValue = isControlled
    ? (val: string) => controlledOnChange?.(val)
    : setInternalValue;

  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value.trim());
      if (clearOnSubmit) {
        setValue("");
      }
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <div className={`flex gap-1.5 ${className}`}>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder={placeholder}
        disabled={disabled}
        className="glass-input flex-1 px-3 py-2 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        className="btn-gradient-orange rounded-xl px-4 py-2 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
      >
        <Icon className="w-4 h-4" />
        {buttonText && <span className="hidden sm:inline">{buttonText}</span>}
      </button>
    </div>
  );
};
