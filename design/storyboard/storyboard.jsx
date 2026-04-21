// AutoStream storyboard scenes — 5 phases of an agent conversation as animated video.
// Uses Stage + Sprite + TextSprite from animations.jsx.

const C = {
  bg: '#0A0A0A',
  bgElev: '#141414',
  line: '#262626',
  fg: '#F5F1EA',
  fgDim: '#A8A29E',
  fgMuted: '#6B6660',
  accent: '#FF5A1F',
  accentDeep: '#D94813',
};

const fontDisplay = '"Space Grotesk", system-ui, sans-serif';
const fontMono = '"JetBrains Mono", ui-monospace, monospace';
const fontBody = '"Inter", system-ui, sans-serif';

// ── Reusable bits ───────────────────────────────────────────────────────────

function ChatFrame({ children, title = 'AUTOSTREAM · LIVE' }) {
  return (
    <div style={{
      position: 'absolute',
      left: 120, top: 120,
      width: 800, height: 840,
      background: C.bgElev,
      border: `1px solid ${C.line}`,
      borderRadius: 18,
      padding: 32,
      display: 'flex', flexDirection: 'column', gap: 16,
      boxShadow: '0 40px 120px rgba(0,0,0,.5)',
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 14,
        color: C.accent, letterSpacing: '0.14em',
        paddingBottom: 16, borderBottom: `1px solid ${C.line}`,
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span>{title}</span>
        <span style={{ color: C.fgMuted }}>●●●</span>
      </div>
      {children}
    </div>
  );
}

function Bubble({ role = 'agent', children, enterAt = 0, delay = 0, localTime }) {
  const t = Math.max(0, (localTime ?? 0) - enterAt - delay);
  const progress = clamp(t / 0.4, 0, 1);
  const eased = Easing.easeOutCubic(progress);
  const isUser = role === 'user';
  return (
    <div style={{
      alignSelf: isUser ? 'flex-end' : 'flex-start',
      maxWidth: '82%',
      padding: '14px 20px',
      borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
      background: isUser ? C.accent : '#1C1C1C',
      color: isUser ? C.bg : C.fg,
      fontSize: 22, lineHeight: 1.4,
      fontWeight: isUser ? 500 : 400,
      fontFamily: fontBody,
      opacity: eased,
      transform: `translateY(${(1 - eased) * 14}px)`,
      willChange: 'opacity, transform',
    }}>
      {children}
    </div>
  );
}

function Chips({ items = [], enterAt = 0, localTime }) {
  const t = Math.max(0, (localTime ?? 0) - enterAt);
  const eased = Easing.easeOutCubic(clamp(t / 0.4, 0, 1));
  return (
    <div style={{
      display: 'flex', gap: 8, flexWrap: 'wrap',
      opacity: eased, transform: `translateY(${(1 - eased) * 10}px)`,
    }}>
      {items.map((label, i) => (
        <span key={i} style={{
          border: `1px solid ${C.accent}`,
          color: C.accent,
          padding: '8px 14px', borderRadius: 999,
          fontFamily: fontMono, fontSize: 16,
        }}>{label}</span>
      ))}
    </div>
  );
}

// Side panel: shows which phase / state we're in
function StatePanel({ phase, intent, slots, running }) {
  return (
    <div style={{
      position: 'absolute',
      right: 120, top: 120,
      width: 740, height: 840,
      background: C.bg, border: `1px solid ${C.line}`,
      borderRadius: 14, padding: 36,
      display: 'flex', flexDirection: 'column', gap: 24,
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 12, color: C.fgMuted,
        letterSpacing: '0.14em', textTransform: 'uppercase',
      }}>[ AGENT STATE ]</div>

      {/* Phase */}
      <div>
        <div style={{ fontFamily: fontMono, fontSize: 13, color: C.fgDim, letterSpacing: '0.12em', marginBottom: 12 }}>PHASE</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['browsing','qualifying','confirming','captured'].map((p, i) => (
            <div key={p} style={{
              flex: 1, padding: '14px 12px',
              border: `1px solid ${phase === p ? C.accent : C.line}`,
              background: phase === p ? 'rgba(255,90,31,.12)' : 'transparent',
              color: phase === p ? C.accent : C.fgDim,
              fontFamily: fontMono, fontSize: 13,
              letterSpacing: '0.08em', textTransform: 'uppercase',
              textAlign: 'center',
              transition: 'all 300ms',
            }}>{String(i+1).padStart(2,'0')} · {p}</div>
          ))}
        </div>
      </div>

      {/* Intent */}
      <div>
        <div style={{ fontFamily: fontMono, fontSize: 13, color: C.fgDim, letterSpacing: '0.12em', marginBottom: 12 }}>LAST INTENT</div>
        <div style={{
          fontFamily: fontMono, fontSize: 20, color: C.fg,
          padding: '12px 16px', background: C.bgElev, border: `1px solid ${C.line}`,
        }}>{intent || '—'}</div>
      </div>

      {/* Slots */}
      <div>
        <div style={{ fontFamily: fontMono, fontSize: 13, color: C.fgDim, letterSpacing: '0.12em', marginBottom: 12 }}>LEAD SLOTS</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { k: 'name', v: slots?.name },
            { k: 'email', v: slots?.email },
            { k: 'platform', v: slots?.platform },
          ].map(({ k, v }) => (
            <div key={k} style={{
              display: 'grid', gridTemplateColumns: '120px 1fr 20px', gap: 14, alignItems: 'center',
              padding: '10px 14px', background: C.bgElev, border: `1px solid ${C.line}`,
            }}>
              <span style={{ fontFamily: fontMono, fontSize: 14, color: C.fgDim, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{k}</span>
              <span style={{ fontFamily: fontMono, fontSize: 18, color: v ? C.fg : C.fgMuted }}>{v || '—'}</span>
              <span style={{ color: v ? C.accent : C.fgMuted, fontFamily: fontMono, fontSize: 18, textAlign: 'right' }}>{v ? '●' : '○'}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Running node */}
      <div style={{ marginTop: 'auto' }}>
        <div style={{ fontFamily: fontMono, fontSize: 13, color: C.fgDim, letterSpacing: '0.12em', marginBottom: 12 }}>LANGGRAPH · EXECUTING</div>
        <div style={{
          fontFamily: fontMono, fontSize: 20,
          padding: '14px 18px', background: '#1a0d06', border: `1px solid ${C.accent}`,
          color: C.accent, letterSpacing: '0.06em',
        }}>
          {running || 'idle'}<span className="blink">▊</span>
        </div>
      </div>
    </div>
  );
}

// Scene title card — shows between scenes
function SceneTitle({ num, title, subtitle }) {
  const { progress, localTime, duration } = useSprite();
  const entry = Easing.easeOutCubic(clamp(localTime / 0.6, 0, 1));
  const exit = duration - localTime < 0.5
    ? 1 - Easing.easeInCubic(clamp(1 - (duration - localTime) / 0.5, 0, 1))
    : 1;
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column', justifyContent: 'center',
      padding: '0 140px',
      opacity: entry * exit,
      background: C.bg,
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 18, color: C.accent,
        letterSpacing: '0.18em', textTransform: 'uppercase',
        marginBottom: 32,
      }}>[ PHASE {num} ]</div>
      <div style={{
        fontFamily: fontDisplay, fontWeight: 600,
        fontSize: 200, letterSpacing: '-0.04em', lineHeight: 0.95,
        color: C.fg,
      }}>{title}</div>
      <div style={{
        fontFamily: fontBody, fontSize: 36, color: C.fgDim,
        marginTop: 40, maxWidth: 1400, lineHeight: 1.4,
      }}>{subtitle}</div>
    </div>
  );
}

// ── Scene: Cover ────────────────────────────────────────────────────────────

function SceneCover() {
  const { localTime, duration } = useSprite();
  const t = Easing.easeOutCubic(clamp(localTime / 0.8, 0, 1));
  const exit = duration - localTime < 0.5
    ? 1 - Easing.easeInCubic(clamp(1 - (duration - localTime) / 0.5, 0, 1))
    : 1;
  return (
    <div style={{
      position: 'absolute', inset: 0,
      background: C.bg,
      display: 'flex', flexDirection: 'column', justifyContent: 'center',
      padding: '0 140px',
      opacity: exit,
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 18, color: C.accent,
        letterSpacing: '0.18em', textTransform: 'uppercase',
        marginBottom: 32, opacity: t, transform: `translateY(${(1-t)*20}px)`,
      }}>[ AUTOSTREAM · LIFE OF A CONVERSATION ]</div>
      <div style={{
        fontFamily: fontDisplay, fontWeight: 600,
        fontSize: 260, letterSpacing: '-0.05em', lineHeight: 0.9,
        color: C.fg,
        opacity: t, transform: `translateY(${(1-t)*30}px)`,
      }}>One visitor.</div>
      <div style={{
        fontFamily: fontDisplay, fontWeight: 600,
        fontSize: 260, letterSpacing: '-0.05em', lineHeight: 0.9,
        color: C.accent,
        opacity: t, transform: `translateY(${(1-t)*40}px)`,
      }}>Five phases.</div>
      <div style={{
        fontFamily: fontBody, fontSize: 36, color: C.fgDim,
        marginTop: 48, maxWidth: 1400, lineHeight: 1.4,
        opacity: t, transform: `translateY(${(1-t)*20}px)`,
      }}>A LangGraph agent turns a stranger into a qualified lead — without a form, a CTA, or a hallucinated price.</div>
    </div>
  );
}

// ── Scene: Browsing (RAG answer) ────────────────────────────────────────────

function SceneBrowsing() {
  const { localTime } = useSprite();
  return (
    <div style={{ position: 'absolute', inset: 0, background: C.bg, display: 'flex' }}>
      <ChatFrame>
        <Bubble role="agent" enterAt={0.2} localTime={localTime}>
          Hey! I'm AutoStream's assistant. Ask me anything about plans, features, or pricing.
        </Bubble>
        <Chips items={['Tell me about pricing', 'Compare plans', 'I want to sign up']} enterAt={0.6} localTime={localTime}/>
        <Bubble role="user" enterAt={2.0} localTime={localTime}>What does Pro include?</Bubble>
        <Bubble role="agent" enterAt={3.4} localTime={localTime}>
          Pro is <b style={{ color: C.accent }}>$79/mo</b> — unlimited edits, 4K export, AI captions, and 24/7 support.
        </Bubble>
        <Chips items={['Sign me up', 'What about Basic?']} enterAt={4.2} localTime={localTime}/>
      </ChatFrame>
      <StatePanel
        phase="browsing"
        intent={localTime > 2.5 ? 'product_inquiry' : 'greeting'}
        slots={{}}
        running={
          localTime < 2.5 ? 'respond' :
          localTime < 3.2 ? 'retrieve · FAISS top-3' :
          'respond'
        }
      />
    </div>
  );
}

// ── Scene: Qualifying (slot collection) ─────────────────────────────────────

function SceneQualifying() {
  const { localTime } = useSprite();
  const slots = {
    name: localTime > 2.8 ? 'Jainam' : '',
    platform: localTime > 6.2 ? 'YouTube' : '',
    email: '',
  };
  return (
    <div style={{ position: 'absolute', inset: 0, background: C.bg, display: 'flex' }}>
      <ChatFrame>
        <Bubble role="user" enterAt={0.2} localTime={localTime}>I want to sign up</Bubble>
        <Bubble role="agent" enterAt={1.2} localTime={localTime}>
          Great! What's your name?
        </Bubble>
        <Bubble role="user" enterAt={2.6} localTime={localTime}>Jainam</Bubble>
        <Bubble role="agent" enterAt={3.8} localTime={localTime}>
          Nice to meet you, Jainam. Which platform do you create for?
        </Bubble>
        <Chips items={['YouTube', 'TikTok', 'Instagram', 'Twitch']} enterAt={4.4} localTime={localTime}/>
        <Bubble role="user" enterAt={6.0} localTime={localTime}>YouTube</Bubble>
      </ChatFrame>
      <StatePanel
        phase="qualifying"
        intent={
          localTime < 1.0 ? 'high_intent' :
          localTime < 3.0 ? 'high_intent' :
          'high_intent'
        }
        slots={slots}
        running={
          localTime < 1.0 ? 'classify_intent' :
          localTime < 3.6 ? 'extract_lead · name' :
          localTime < 6.6 ? 'extract_lead · platform' :
          'respond'
        }
      />
    </div>
  );
}

// ── Scene: Confirming ───────────────────────────────────────────────────────

function SceneConfirming() {
  const { localTime } = useSprite();
  const slots = {
    name: 'Jainam',
    platform: 'YouTube',
    email: localTime > 3.0 ? 'jainam@demo.app' : '',
  };
  return (
    <div style={{ position: 'absolute', inset: 0, background: C.bg, display: 'flex' }}>
      <ChatFrame>
        <Bubble role="agent" enterAt={0.2} localTime={localTime}>
          And what email should I send your onboarding to?
        </Bubble>
        <Bubble role="user" enterAt={2.4} localTime={localTime}>jainam@demo.app</Bubble>
        <Bubble role="agent" enterAt={4.0} localTime={localTime}>
          Quick recap:<br/>
          <b style={{ color: C.accent }}>Jainam · jainam@demo.app · YouTube.</b><br/>
          Ready to submit?
        </Bubble>
        <Chips items={['Submit', 'Fix something']} enterAt={5.0} localTime={localTime}/>
      </ChatFrame>
      <StatePanel
        phase={localTime > 3.0 ? 'confirming' : 'qualifying'}
        intent="high_intent"
        slots={slots}
        running={
          localTime < 2.8 ? 'respond' :
          localTime < 3.8 ? 'extract_lead · email · regex ✓' :
          'respond · recap'
        }
      />
    </div>
  );
}

// ── Scene: Captured (tool fires) ────────────────────────────────────────────

function SceneCaptured() {
  const { localTime } = useSprite();
  const slots = { name: 'Jainam', platform: 'YouTube', email: 'jainam@demo.app' };

  // Animate the tool-call flash
  const flashT = clamp((localTime - 2.0) / 0.6, 0, 1);
  const flash = flashT > 0 && flashT < 1 ? Math.sin(flashT * Math.PI) : 0;

  return (
    <div style={{ position: 'absolute', inset: 0, background: C.bg, display: 'flex' }}>
      <ChatFrame>
        <Bubble role="user" enterAt={0.2} localTime={localTime}>Submit</Bubble>
        <Bubble role="agent" enterAt={3.0} localTime={localTime}>
          All set, Jainam! 🎬 You'll hear from us within 24 hours.<br/>
          Meanwhile, anything else you'd like to know?
        </Bubble>
        <Chips items={['Tell me about refunds', 'How does auto-cut work?']} enterAt={3.8} localTime={localTime}/>
      </ChatFrame>

      {/* Tool fire indicator */}
      {localTime > 1.6 && localTime < 3.0 && (
        <div style={{
          position: 'absolute',
          right: 120, bottom: 120,
          padding: '20px 32px',
          background: C.accent, color: C.bg,
          border: `2px solid ${C.accent}`,
          borderRadius: 8,
          fontFamily: fontMono, fontSize: 22,
          letterSpacing: '0.06em',
          transform: `scale(${1 + flash * 0.08})`,
          boxShadow: `0 0 ${40 + flash * 40}px rgba(255,90,31,.6)`,
          fontWeight: 600,
        }}>
          ▶ mock_lead_capture() fired
        </div>
      )}

      <StatePanel
        phase={localTime < 1.6 ? 'confirming' : localTime < 4 ? 'captured' : 'browsing'}
        intent={localTime < 1.6 ? 'high_intent' : 'product_inquiry'}
        slots={slots}
        running={
          localTime < 1.4 ? 'classify_intent' :
          localTime < 2.4 ? 'capture · tool_call' :
          'respond'
        }
      />
    </div>
  );
}

// ── Scene: Close / tagline ──────────────────────────────────────────────────

function SceneClose() {
  const { localTime, duration } = useSprite();
  const t = Easing.easeOutCubic(clamp(localTime / 0.8, 0, 1));
  return (
    <div style={{
      position: 'absolute', inset: 0,
      background: C.accent, color: C.bg,
      display: 'flex', flexDirection: 'column', justifyContent: 'center',
      padding: '0 140px',
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 18, color: C.bg,
        letterSpacing: '0.18em', textTransform: 'uppercase',
        marginBottom: 32, opacity: t,
      }}>[ AUTOSTREAM ]</div>
      <div style={{
        fontFamily: fontDisplay, fontWeight: 600,
        fontSize: 280, letterSpacing: '-0.05em', lineHeight: 0.9,
        opacity: t, transform: `translateY(${(1-t)*30}px)`,
      }}>Ship videos.</div>
      <div style={{
        fontFamily: fontDisplay, fontWeight: 600,
        fontSize: 280, letterSpacing: '-0.05em', lineHeight: 0.9,
        textDecoration: 'line-through',
        textDecorationColor: C.bg,
        textDecorationThickness: 12,
        color: 'rgba(10,10,10,.4)',
        opacity: t, transform: `translateY(${(1-t)*40}px)`,
      }}>Not edits.</div>
      <div style={{
        fontFamily: fontMono, fontSize: 22, color: C.bg,
        letterSpacing: '0.14em', marginTop: 64,
        opacity: t,
      }}>autostream.app · built with langgraph + groq + faiss</div>
    </div>
  );
}

// ── Timeline ────────────────────────────────────────────────────────────────
// Total: 42 seconds

function Storyboard() {
  return (
    <>
      <Sprite start={0} end={4}><SceneCover /></Sprite>

      <Sprite start={4} end={6.5}>
        <SceneTitle num="01" title="Browsing."
          subtitle="Visitor asks a product question. RAG retrieves. Agent answers — without hallucinating."/>
      </Sprite>
      <Sprite start={6.5} end={12.5}><SceneBrowsing /></Sprite>

      <Sprite start={12.5} end={15}>
        <SceneTitle num="02" title="Qualifying."
          subtitle="High intent detected. Phase flips. Slots get filled one at a time. Free text always."/>
      </Sprite>
      <Sprite start={15} end={23}><SceneQualifying /></Sprite>

      <Sprite start={23} end={25.5}>
        <SceneTitle num="03" title="Confirming."
          subtitle="Email regex passes. Recap presented. User must explicitly confirm. Tool can't fire yet."/>
      </Sprite>
      <Sprite start={25.5} end={32}><SceneConfirming /></Sprite>

      <Sprite start={32} end={34.5}>
        <SceneTitle num="04" title="Captured."
          subtitle="mock_lead_capture() fires once. Phase drops to browsing. Follow-ups route normally."/>
      </Sprite>
      <Sprite start={34.5} end={40}><SceneCaptured /></Sprite>

      <Sprite start={40} end={45}><SceneClose /></Sprite>
    </>
  );
}

function StoryboardApp() {
  return (
    <Stage width={1920} height={1080} duration={45} background={C.bg} persistKey="autostream-storyboard">
      <Storyboard />
    </Stage>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<StoryboardApp />);
