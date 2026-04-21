// Custom web chat widget — AutoStream branded floating panel
// Bold, dark, punchy. Big display font, orange accent, monospace chips.

function Avatar({ role, size = 28 }) {
  if (role === 'user') {
    return (
      <div style={{
        width: size, height: size, borderRadius: size/2,
        background: 'linear-gradient(140deg, #3a3a3a, #1a1a1a)',
        border: '1px solid #333',
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#A8A29E', fontFamily: 'JetBrains Mono, monospace',
        fontSize: 11, fontWeight: 600, letterSpacing: '0.04em',
      }}>J</div>
    );
  }
  return (
    <div style={{
      width: size, height: size, borderRadius: 4,
      background: '#FF5A1F', flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      boxShadow: '0 0 0 1px rgba(255,255,255,.08), 0 4px 14px rgba(255,90,31,.35)',
    }}>
      <svg width={size*0.55} height={size*0.55} viewBox="0 0 20 20" fill="none">
        <rect x="3" y="6" width="10" height="8" rx="1" fill="#0A0A0A"/>
        <path d="M13 9 L17 6 V14 L13 11 Z" fill="#0A0A0A"/>
      </svg>
    </div>
  );
}

function AgentBubble({ text, showTypingDots = false }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', maxWidth: '88%' }}>
      <Avatar role="agent" />
      <div style={{
        background: '#141414',
        border: '1px solid #262626',
        color: '#F5F1EA',
        padding: '10px 14px',
        borderRadius: '4px 14px 14px 14px',
        fontSize: 14, lineHeight: 1.5,
        fontFamily: 'Inter, sans-serif',
        letterSpacing: '-0.005em',
      }}>
        {showTypingDots ? (
          <span style={{ display: 'inline-flex', gap: 4, padding: '4px 0' }}>
            <i style={{ width: 5, height: 5, borderRadius: 3, background: '#FF5A1F', display: 'inline-block' }} />
            <i style={{ width: 5, height: 5, borderRadius: 3, background: '#FF5A1F', opacity: .6, display: 'inline-block' }} />
            <i style={{ width: 5, height: 5, borderRadius: 3, background: '#FF5A1F', opacity: .3, display: 'inline-block' }} />
          </span>
        ) : text}
      </div>
    </div>
  );
}

function UserBubble({ text }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', maxWidth: '88%', alignSelf: 'flex-end', marginLeft: 'auto', flexDirection: 'row-reverse' }}>
      <Avatar role="user" />
      <div style={{
        background: '#FF5A1F',
        color: '#0A0A0A',
        padding: '10px 14px',
        borderRadius: '14px 4px 14px 14px',
        fontSize: 14, lineHeight: 1.5,
        fontFamily: 'Inter, sans-serif',
        fontWeight: 500,
        letterSpacing: '-0.005em',
      }}>{text}</div>
    </div>
  );
}

function QuickReplies({ labels, onPick }) {
  if (!labels || !labels.length) return null;
  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', gap: 6,
      marginLeft: 38, marginTop: -4, marginBottom: 4,
    }}>
      {labels.map((l, i) => (
        <button key={i} onClick={() => onPick && onPick(l)}
          style={{
            background: '#0A0A0A',
            border: '1px solid #FF5A1F',
            color: '#FF5A1F',
            padding: '5px 11px',
            borderRadius: 999,
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 11, letterSpacing: '0.02em',
            cursor: 'pointer',
            transition: 'all .15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = '#FF5A1F'; e.currentTarget.style.color = '#0A0A0A'; }}
          onMouseLeave={e => { e.currentTarget.style.background = '#0A0A0A'; e.currentTarget.style.color = '#FF5A1F'; }}
        >{l}</button>
      ))}
    </div>
  );
}

function PhaseBadge({ phase }) {
  const colors = {
    browsing: { fg: '#A8A29E', dot: '#A8A29E' },
    qualifying: { fg: '#FFB020', dot: '#FFB020' },
    confirming: { fg: '#7CF29C', dot: '#7CF29C' },
    captured: { fg: '#FF5A1F', dot: '#FF5A1F' },
  };
  const c = colors[phase] || colors.browsing;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase',
      color: c.fg,
    }}>
      <i style={{ width: 5, height: 5, borderRadius: 3, background: c.dot, boxShadow: `0 0 0 3px ${c.dot}22` }} />
      {phase}
    </span>
  );
}

function WebChatWidget({ width = 380, height = 620, upThroughIndex = 11, showHeader = true, showStateRail = true, title = 'AutoStream' }) {
  const msgs = window.CONVERSATION_SAMPLE.slice(0, upThroughIndex + 1);
  const last = msgs[msgs.length - 1];
  const lastReplies = last && last.role === 'agent' ? last.quick_replies : null;

  // Determine current phase based on CONVERSATION_PHASES
  let phase = 'browsing';
  let slots = { name: null, email: null, platform: null };
  for (const p of window.CONVERSATION_PHASES) {
    if (p.after <= upThroughIndex) {
      phase = p.phase;
      if (p.slots) slots = { ...slots, ...p.slots };
    }
  }

  return (
    <div style={{
      width, height,
      background: '#0A0A0A',
      border: '1px solid #262626',
      borderRadius: 16,
      overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
      fontFamily: 'Inter, sans-serif',
      boxShadow: '0 40px 120px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.02) inset',
    }}>
      {/* Header */}
      {showHeader && (
        <div style={{
          padding: '14px 16px',
          borderBottom: '1px solid #262626',
          display: 'flex', alignItems: 'center', gap: 12,
          background: 'linear-gradient(180deg, #141414, #0A0A0A)',
        }}>
          <Avatar role="agent" size={32} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontFamily: 'Space Grotesk, sans-serif', fontWeight: 600, fontSize: 15, color: '#F5F1EA', letterSpacing: '-0.01em' }}>
              {title}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <i style={{ width: 6, height: 6, borderRadius: 3, background: '#7CF29C', boxShadow: '0 0 0 3px rgba(124,242,156,.15)' }} />
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#A8A29E', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Online · replies in ~1s
              </span>
            </div>
          </div>
          <button style={{
            background: 'transparent', border: 'none', color: '#6B6660',
            cursor: 'pointer', padding: 4,
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
          </button>
        </div>
      )}

      {/* State rail */}
      {showStateRail && (
        <div style={{
          padding: '8px 16px',
          background: '#0F0F0F',
          borderBottom: '1px solid #1F1F1F',
          display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap',
        }}>
          <PhaseBadge phase={phase} />
          <div style={{ width: 1, height: 12, background: '#262626' }} />
          <div style={{ display: 'flex', gap: 8, fontFamily: 'JetBrains Mono, monospace', fontSize: 10, letterSpacing: '0.04em' }}>
            {['name','email','platform'].map(k => (
              <span key={k} style={{ color: slots[k] ? '#F5F1EA' : '#6B6660', textTransform: 'uppercase' }}>
                {slots[k] ? '●' : '○'} {k}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div style={{
        flex: 1, overflow: 'auto',
        padding: '16px 14px',
        display: 'flex', flexDirection: 'column', gap: 10,
        background: 'radial-gradient(1200px 400px at 50% -10%, rgba(255,90,31,.04), transparent)',
      }}>
        {msgs.map((m, i) => (
          <React.Fragment key={i}>
            {m.role === 'agent' ? <AgentBubble text={m.text} /> : <UserBubble text={m.text} />}
            {i === msgs.length - 1 && m.role === 'agent' && lastReplies && lastReplies.length > 0 && (
              <QuickReplies labels={lastReplies} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Composer */}
      <div style={{
        padding: '10px 12px',
        borderTop: '1px solid #262626',
        background: '#0F0F0F',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <div style={{
          flex: 1, background: '#1C1C1C', border: '1px solid #262626',
          borderRadius: 10, padding: '8px 12px', fontSize: 13,
          color: '#6B6660', fontFamily: 'Inter, sans-serif',
        }}>Type anything — or tap a suggestion above</div>
        <button style={{
          width: 36, height: 36, borderRadius: 10,
          background: '#FF5A1F', border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 4px 14px rgba(255,90,31,.35)',
        }}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 7h12M8 2l5 5-5 5" stroke="#0A0A0A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  );
}

Object.assign(window, { WebChatWidget, AgentBubble, UserBubble, QuickReplies, PhaseBadge, Avatar });
