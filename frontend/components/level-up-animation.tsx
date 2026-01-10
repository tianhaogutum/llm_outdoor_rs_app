"use client";

import React, { useEffect, useState } from "react";
import Confetti from "react-confetti";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, Star, Sparkles, Zap } from "lucide-react";

interface LevelUpAnimationProps {
  newLevel: number;
  isVisible: boolean;
  onComplete?: () => void;
}

export function LevelUpAnimation({
  newLevel,
  isVisible,
  onComplete,
}: LevelUpAnimationProps) {
  const [showConfetti, setShowConfetti] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    // Set window size for confetti
    if (typeof window !== "undefined") {
      setWindowSize({ width: window.innerWidth, height: window.innerHeight });

      const handleResize = () => {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      };
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }
  }, []);

  useEffect(() => {
    if (isVisible) {
      setShowConfetti(true);

      // Stop confetti after 4 seconds
      const confettiTimer = setTimeout(() => {
        setShowConfetti(false);
      }, 4000);

      // Auto-complete animation after 5 seconds
      const completeTimer = setTimeout(() => {
        if (onComplete) onComplete();
      }, 5000);

      return () => {
        clearTimeout(confettiTimer);
        clearTimeout(completeTimer);
      };
    }
  }, [isVisible, onComplete]);

  return (
    <AnimatePresence>
      {isVisible && (
        <>
          {/* Confetti */}
          {showConfetti && (
            <Confetti
              width={windowSize.width}
              height={windowSize.height}
              recycle={false}
              numberOfPieces={500}
              gravity={0.3}
              colors={[
                "#FFD700",
                "#FFA500",
                "#FF69B4",
                "#00CED1",
                "#9370DB",
                "#32CD32",
              ]}
            />
          )}

          {/* Full screen overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={onComplete}
          >
            {/* Background animated particles */}
            <div className="absolute inset-0 overflow-hidden">
              {[...Array(20)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-2 h-2 bg-yellow-400 rounded-full"
                  initial={{
                    x: Math.random() * 100 + "%",
                    y: "100%",
                    opacity: 0,
                  }}
                  animate={{
                    y: "-20%",
                    opacity: [0, 1, 1, 0],
                    scale: [0, 1.5, 1, 0],
                  }}
                  transition={{
                    duration: 2 + Math.random() * 2,
                    repeat: Infinity,
                    delay: Math.random() * 2,
                    ease: "easeOut",
                  }}
                />
              ))}
            </div>

            {/* Main content */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              exit={{ scale: 0, rotate: 180 }}
              transition={{
                type: "spring",
                stiffness: 200,
                damping: 20,
                duration: 0.8,
              }}
              className="relative z-10"
            >
              <div className="text-center px-8 py-12 bg-gradient-to-b from-purple-900/90 to-indigo-900/90 rounded-3xl border-4 border-yellow-400 shadow-2xl">
                {/* Floating stars around */}
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute"
                    style={{
                      left: `${50 + 40 * Math.cos((i * Math.PI * 2) / 8)}%`,
                      top: `${50 + 40 * Math.sin((i * Math.PI * 2) / 8)}%`,
                    }}
                    animate={{
                      scale: [1, 1.5, 1],
                      rotate: [0, 360],
                      opacity: [1, 0.5, 1],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                  >
                    <Star className="w-6 h-6 text-yellow-400 fill-yellow-400" />
                  </motion.div>
                ))}

                {/* Level Up Text */}
                <motion.div
                  initial={{ y: -50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="mb-6"
                >
                  <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 via-orange-500 to-pink-500 mb-2">
                    LEVEL UP!
                  </h1>
                  <div className="flex items-center justify-center gap-2">
                    <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
                    <p className="text-2xl text-yellow-200 font-semibold">
                      Congratulations!
                    </p>
                    <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
                  </div>
                </motion.div>

                {/* Level number with pulsing animation */}
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: [0, 1.2, 1] }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                  className="relative mb-6"
                >
                  {/* Glow effect */}
                  <motion.div
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 0.8, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                    }}
                    className="absolute inset-0 bg-yellow-400 blur-3xl opacity-50"
                  />

                  {/* Level badge */}
                  <div className="relative bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full w-40 h-40 mx-auto flex items-center justify-center border-8 border-yellow-300 shadow-2xl">
                    <div className="text-center">
                      <Trophy className="w-12 h-12 text-white mx-auto mb-2" />
                      <p className="text-white text-2xl font-bold">LEVEL</p>
                      <p className="text-white text-5xl font-black">
                        {newLevel}
                      </p>
                    </div>
                  </div>
                </motion.div>

                {/* Lightning bolts */}
                <div className="flex items-center justify-center gap-4 mb-6">
                  <motion.div
                    animate={{
                      rotate: [0, 10, -10, 0],
                      scale: [1, 1.2, 1],
                    }}
                    transition={{
                      duration: 0.5,
                      repeat: Infinity,
                      repeatDelay: 1,
                    }}
                  >
                    <Zap className="w-8 h-8 text-yellow-400 fill-yellow-400" />
                  </motion.div>
                  <motion.div
                    animate={{
                      rotate: [0, -10, 10, 0],
                      scale: [1, 1.2, 1],
                    }}
                    transition={{
                      duration: 0.5,
                      repeat: Infinity,
                      repeatDelay: 1,
                      delay: 0.2,
                    }}
                  >
                    <Zap className="w-8 h-8 text-yellow-400 fill-yellow-400" />
                  </motion.div>
                </div>

                {/* Message */}
                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.8 }}
                >
                  <p className="text-white text-xl font-semibold mb-2">
                    You've reached Level {newLevel}!
                  </p>
                  <p className="text-yellow-200 text-sm">
                    Keep exploring to unlock more adventures
                  </p>
                </motion.div>

                {/* Tap to continue hint */}
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 2 }}
                  className="text-gray-300 text-xs mt-6"
                >
                  Tap anywhere to continue
                </motion.p>
              </div>
            </motion.div>

            {/* Sparkle effects */}
            {[...Array(15)].map((_, i) => (
              <motion.div
                key={`sparkle-${i}`}
                className="absolute"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{
                  scale: [0, 1, 0],
                  opacity: [0, 1, 0],
                  rotate: [0, 180, 360],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  delay: Math.random() * 2,
                  repeatDelay: Math.random() * 2,
                }}
              >
                <Sparkles className="w-4 h-4 text-yellow-400" />
              </motion.div>
            ))}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
