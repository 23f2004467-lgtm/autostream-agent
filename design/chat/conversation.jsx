// Shared chat conversation data — the sample conversation from the plan §13,
// plus a few extended variants for demonstrating different states.
// Each message has: role ('user' | 'agent'), text, and optional quick_replies (for agent).

const CONVERSATION_SAMPLE = [
  { role: 'agent', text: "Hey! I'm AutoStream's assistant. Want to hear about our plans, features, or pricing?", quick_replies: ["Tell me about pricing", "Compare plans", "I want to sign up"] },
  { role: 'user', text: "What does the Pro plan include?" },
  { role: 'agent', text: "The Pro Plan is $79/month and includes unlimited video edits, 4K export, AI captions, and 24/7 customer support — built for full-time creators.", quick_replies: ["Sign me up", "Compare plans", "What about refunds?"] },
  { role: 'user', text: "That sounds great, I want to sign up for my YouTube channel." },
  { role: 'agent', text: "Love that! I just need a few details to get you set up. What's your name?", quick_replies: [] },
  { role: 'user', text: "Jainam" },
  { role: 'agent', text: "Nice to meet you, Jainam. What's the best email to reach you?", quick_replies: [] },
  { role: 'user', text: "jainam@distill.fyi" },
  { role: 'agent', text: "Got it. Just to confirm — Jainam, jainam@distill.fyi, YouTube. Ready to submit?", quick_replies: ["Yes, submit", "Fix something"] },
  { role: 'user', text: "Yes, submit" },
  { role: 'agent', text: "All set, Jainam — we've captured your details and someone from the AutoStream team will reach out shortly.", quick_replies: ["Ask another question", "Talk to sales", "I'm good, thanks"] },
];

// Conversation state annotation for the widget — shows phase transitions
// as chips in the UI so reviewers can trace the state machine.
const CONVERSATION_PHASES = [
  { after: 0, phase: 'browsing', intent: 'greeting' },
  { after: 2, phase: 'browsing', intent: 'product_inquiry' },
  { after: 4, phase: 'qualifying', intent: 'high_intent', slots: { platform: 'YouTube' } },
  { after: 6, phase: 'qualifying', slots: { name: 'Jainam', platform: 'YouTube' } },
  { after: 8, phase: 'confirming', slots: { name: 'Jainam', email: 'jainam@distill.fyi', platform: 'YouTube' } },
  { after: 10, phase: 'captured', slots: { name: 'Jainam', email: 'jainam@distill.fyi', platform: 'YouTube' } },
];

Object.assign(window, { CONVERSATION_SAMPLE, CONVERSATION_PHASES });
