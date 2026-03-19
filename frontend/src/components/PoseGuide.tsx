"use client";

import React, { useState, useEffect } from "react";

interface PoseGuideProps {
  exerciseName: string;
  poseImageSrc: string | string[];
  instructions: string[];
  onStart: () => void;
  onBack?: () => void;
}

export default function PoseGuide({
  exerciseName,
  poseImageSrc,
  instructions,
  onStart,
  onBack,
}: PoseGuideProps) {
  const [countdown, setCountdown] = useState<number | null>(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown !== null && countdown > 0) {
      // Trigger scale pulse effect on each tick
      setScale(1.2);
      const scaleTimeout = setTimeout(() => setScale(1), 200);
      
      timer = setTimeout(() => {
        setCountdown((prev) => (prev !== null ? prev - 1 : null));
      }, 1000);

      return () => {
        clearTimeout(timer);
        clearTimeout(scaleTimeout);
      };
    } else if (countdown === 0) {
      // Small delay before firing onStart so the user can see "1" or "GO"
      timer = setTimeout(() => {
        onStart();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [countdown, onStart]);

  const handleStart = () => {
    setCountdown(5);
  };

  // Design system colors & typography
  const bgMain = "#f7f9f7";
  const greenAccent = "#1a6640";
  const borderGreen = "#e3ede5";
  const white = "#ffffff";
  const textDark = "#1a1a1a";
  const textLight = "#666666";

  const fontHeading = '"Fraunces", serif';
  const fontBody = '"DM Sans", sans-serif';

  if (countdown !== null) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          backgroundColor: bgMain,
          fontFamily: fontBody,
        }}
      >
        <div
          style={{
            fontSize: "10rem",
            color: greenAccent,
            fontFamily: fontHeading,
            fontWeight: "bold",
            transform: `scale(${scale})`,
            transition: "transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
          }}
        >
          {countdown > 0 ? countdown : "1"}
        </div>
        <p
          style={{
            fontSize: "1.5rem",
            color: textLight,
            marginTop: "20px",
            fontFamily: fontBody,
            opacity: scale > 1 ? 0.8 : 1,
            transition: "opacity 0.2s",
          }}
        >
          Get ready...
        </p>
      </div>
    );
  }

  return (
    <div
      style={{
        backgroundColor: bgMain,
        minHeight: "100vh",
        fontFamily: fontBody,
        color: textDark,
        padding: "20px 40px",
      }}
    >
      {/* Navbar */}
      <nav
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "40px",
        }}
      >
        <button
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: "1rem",
            color: greenAccent,
            fontWeight: "500",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <span style={{ fontSize: "1.2rem" }}>&larr;</span> Back
        </button>
        <div
          style={{
            fontFamily: fontHeading,
            fontSize: "1.5rem",
            fontWeight: "bold",
            color: greenAccent,
          }}
        >
          PoseCorrect
        </div>
        <div style={{ width: "70px" }} /> {/* Spacer to align title to center */}
      </nav>

      {/* Header */}
      <h1
        style={{
          fontFamily: fontHeading,
          fontSize: "2.5rem",
          fontWeight: "bold",
          marginBottom: "8px",
          textAlign: "center",
          color: greenAccent,
          marginTop: 0,
        }}
      >
        {exerciseName}
      </h1>
      <p
        style={{
          textAlign: "center",
          fontSize: "1.1rem",
          color: textLight,
          marginBottom: "40px",
          marginTop: 0,
        }}
      >
        Get into position before we start
      </p>

      {/* Two Columns */}
      <div
        style={{
          display: "flex",
          gap: "30px",
          maxWidth: "1000px",
          margin: "0 auto 40px auto",
          flexWrap: "wrap",
        }}
      >
        {/* Left Column - Image Container */}
        <div
          style={{
            flex: "1 1 400px",
            backgroundColor: white,
            border: `1px solid ${borderGreen}`,
            borderRadius: "16px",
            padding: "24px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <div
            style={{
              width: "100%",
              height: "450px",
              position: "relative",
              borderRadius: "12px",
              overflow: "hidden",
              marginBottom: "16px",
              backgroundColor: "#f0f0f0",
              boxShadow: "inset 0 2px 10px rgba(0,0,0,0.03)",
            }}
          >
            {Array.isArray(poseImageSrc) && poseImageSrc.length > 0 ? (
              <div style={{ display: 'flex', width: '100%', height: '100%', gap: '4px' }}>
                {poseImageSrc.map((src, idx) => (
                  <img
                    key={idx}
                    src={src}
                    alt={`${exerciseName} Starting Position ${idx + 1}`}
                    style={{ flex: 1, height: '100%', objectFit: 'cover', minWidth: 0 }}
                  />
                ))}
              </div>
            ) : poseImageSrc && typeof poseImageSrc === "string" ? (
              <img
                src={poseImageSrc}
                alt={`${exerciseName} Starting Position`}
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            ) : (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  color: textLight,
                }}
              >
                Pose Image Placeholder
              </div>
            )}
          </div>
          <p
            style={{
              fontSize: "1.1rem",
              fontWeight: "600",
              color: greenAccent,
              margin: 0,
            }}
          >
            Starting position
          </p>
        </div>

        {/* Right Column - Instructions */}
        <div
          style={{
            flex: "1 1 400px",
            backgroundColor: white,
            border: `1px solid ${borderGreen}`,
            borderRadius: "16px",
            padding: "32px",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <h2
            style={{
              fontFamily: fontHeading,
              fontSize: "1.5rem",
              marginBottom: "24px",
              marginTop: 0,
              color: textDark,
            }}
          >
            Instructions
          </h2>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              flex: 1,
            }}
          >
            {instructions.map((inst, index) => (
              <div
                key={index}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "16px",
                  padding: "16px",
                  backgroundColor: bgMain,
                  borderRadius: "12px",
                  border: `1px solid ${borderGreen}`,
                }}
              >
                <div
                  style={{
                    width: "36px",
                    height: "36px",
                    borderRadius: "50%",
                    backgroundColor: greenAccent,
                    color: white,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: "bold",
                    flexShrink: 0,
                    fontSize: "1.1rem",
                  }}
                >
                  {index + 1}
                </div>
                <span
                  style={{
                    fontSize: "1.05rem",
                    lineHeight: "1.5",
                    color: textDark,
                  }}
                >
                  {inst}
                </span>
              </div>
            ))}
          </div>

          {/* Camera Tip */}
          <div
            style={{
              marginTop: "30px",
              padding: "16px 20px",
              backgroundColor: "rgba(26, 102, 64, 0.06)",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              gap: "14px",
            }}
          >
            <span style={{ fontSize: "1.5rem" }}>📷</span>
            <p
              style={{
                margin: 0,
                fontSize: "1rem",
                color: greenAccent,
                fontWeight: "500",
                lineHeight: "1.4",
              }}
            >
              Make sure your full upper body is visible in the camera
            </p>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div style={{ textAlign: "center", paddingBottom: "60px" }}>
        <button
          onClick={handleStart}
          style={{
            backgroundColor: greenAccent,
            color: white,
            border: "none",
            borderRadius: "10px",
            padding: "18px 56px",
            fontSize: "1.25rem",
            fontWeight: "bold",
            cursor: "pointer",
            fontFamily: fontBody,
            boxShadow: "0 8px 16px rgba(26, 102, 64, 0.2)",
            transition: "transform 0.1s ease-in-out, background-color 0.2s",
          }}
          onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
          onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.backgroundColor = greenAccent;
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.backgroundColor = "#134e30")
          }
        >
          I am ready
        </button>
      </div>
    </div>
  );
}
