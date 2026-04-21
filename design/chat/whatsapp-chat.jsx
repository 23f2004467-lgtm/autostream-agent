// WhatsApp iOS conversation view — tailored to match WA visual language
// but with AutoStream branding. Inside an iOS device frame.

function WABubble({ role, text, time, showTail = true }) {
  const isUser = role === 'user';
  return (
    <div style={{
      display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start',
      padding: '1px 8px', position: 'relative',
    }}>
      <div style={{
        maxWidth: '78%',
        background: isUser ? '#E7FFDB' : '#FFFFFF',
        color: '#0B1418',
        padding: '6px 9px 8px',
        borderRadius: 7.5,
        fontSize: 15.5, lineHeight: 1.32,
        fontFamily: '-apple-system, system-ui',
        boxShadow: '0 1px 0.5px rgba(11,20,26,.13)',
        position: 'relative',
        letterSpacing: '-0.01em',
      }}>
        <div style={{ paddingRight: 54, whiteSpace: 'pre-wrap' }}>{text}</div>
        <div style={{
          position: 'absolute', bottom: 4, right: 8,
          fontSize: 11, color: '#667781', lineHeight: 1,
          display: 'flex', alignItems: 'center', gap: 3,
        }}>
          {time}
          {isUser && (
            <svg width="16" height="11" viewBox="0 0 16 11" fill="none">
              <path d="M11.071 0.653L12.12 1.5 6.32 7.32l-.85-.68 5.6-5.99zM5.6 6.32l1 .77 5.62-5.6-.85-.84L5.6 6.32zM1 5.5l3.1 3.1L10.5 2.2" stroke="#53BDEB" strokeWidth="0.8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}

// Whatsapp interactive buttons — rendered as a separate message with buttons below
function WAButtonMessage({ text, buttons, time }) {
  return (
    <div style={{ padding: '1px 8px', display: 'flex' }}>
      <div style={{
        maxWidth: '78%',
        background: '#FFFFFF',
        color: '#0B1418',
        padding: '6px 9px 8px',
        borderRadius: 7.5,
        fontSize: 15.5, lineHeight: 1.32,
        fontFamily: '-apple-system, system-ui',
        boxShadow: '0 1px 0.5px rgba(11,20,26,.13)',
        letterSpacing: '-0.01em',
      }}>
        <div style={{ paddingRight: 54, position: 'relative' }}>
          {text}
          <div style={{
            position: 'absolute', bottom: -4, right: 0,
            fontSize: 11, color: '#667781',
          }}>{time}</div>
        </div>
        {/* divider + buttons */}
        <div style={{ marginTop: 12, marginLeft: -9, marginRight: -9, marginBottom: -8 }}>
          {buttons.map((b, i) => (
            <div key={i} style={{
              borderTop: '1px solid rgba(11,20,26,.08)',
              padding: '10px 9px',
              textAlign: 'center',
              color: '#027EB5', fontSize: 14.5, fontWeight: 500,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            }}>
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <path d="M10 3h-7a1 1 0 00-1 1v5a1 1 0 001 1h2l2 2v-2h3a1 1 0 001-1V4a1 1 0 00-1-1z" stroke="#027EB5" strokeWidth="1.2" fill="none"/>
              </svg>
              {b}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function WAHeader() {
  return (
    <div style={{
      background: '#F6F5F3',
      padding: '10px 12px 10px 8px',
      display: 'flex', alignItems: 'center', gap: 10,
      borderBottom: '1px solid rgba(0,0,0,.08)',
      position: 'relative',
    }}>
      <div style={{ color: '#007AFF', fontSize: 28, lineHeight: 1, paddingTop: 2 }}>‹</div>
      <div style={{
        width: 36, height: 36, borderRadius: 18,
        background: '#FF5A1F', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 2px 8px rgba(255,90,31,.3)',
      }}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <rect x="3" y="6" width="10" height="8" rx="1" fill="#0A0A0A"/>
          <path d="M13 9 L17 6 V14 L13 11 Z" fill="#0A0A0A"/>
        </svg>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 17, fontWeight: 600, color: '#0B1418', letterSpacing: '-0.02em', fontFamily: '-apple-system' }}>
          AutoStream
        </div>
        <div style={{ fontSize: 12.5, color: '#667781', lineHeight: 1.2 }}>
          online
        </div>
      </div>
      <div style={{ display: 'flex', gap: 18, color: '#007AFF' }}>
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M15 7l5-2v12l-5-2M2 6h13v10H2z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round"/></svg>
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none"><path d="M5 4.5c0 9 4 13 13 13l2-4-4-2-2 2c-3 0-6-3-6-6l2-2-2-4-3 3z" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinejoin="round"/></svg>
      </div>
    </div>
  );
}

function WACompose() {
  return (
    <div style={{
      background: '#F6F5F3',
      padding: '8px 10px',
      display: 'flex', alignItems: 'center', gap: 8,
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 16,
        background: '#FFF', border: '1px solid rgba(0,0,0,.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#667781', fontSize: 22, lineHeight: 1,
      }}>+</div>
      <div style={{
        flex: 1, background: '#FFF',
        border: '1px solid rgba(0,0,0,.1)',
        borderRadius: 20, padding: '6px 12px',
        fontSize: 15.5, color: '#667781',
        fontFamily: '-apple-system',
      }}>Message</div>
      <div style={{
        width: 32, height: 32, borderRadius: 16,
        background: '#25D366', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M9 1v10m0 0l-4-4m4 4l4-4M3 13v3h12v-3" stroke="#FFF" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" transform="rotate(180 9 9)"/>
        </svg>
      </div>
    </div>
  );
}

function WhatsAppChat({ upThroughIndex = 8 }) {
  // Build bubbles but include the quick-reply buttons as WA-interactive messages
  const msgs = [];
  const full = window.CONVERSATION_SAMPLE.slice(0, upThroughIndex + 1);
  let baseTime = 10 * 60 + 23; // 10:23
  for (let i = 0; i < full.length; i++) {
    const m = full[i];
    const h = Math.floor(baseTime / 60);
    const mm = String(baseTime % 60).padStart(2, '0');
    const timeStr = `${h}:${mm}`;
    if (m.role === 'agent' && m.quick_replies && m.quick_replies.length > 0 && m.quick_replies.length <= 3) {
      msgs.push(<WAButtonMessage key={i} text={m.text} buttons={m.quick_replies} time={timeStr} />);
    } else {
      msgs.push(<WABubble key={i} role={m.role} text={m.text} time={timeStr} />);
    }
    baseTime += 1;
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', width: '100%',
      background: '#EFE7DE',
      backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cg fill='rgba(0,0,0,0.03)'%3E%3Ccircle cx='20' cy='30' r='2'/%3E%3Ccircle cx='80' cy='60' r='2'/%3E%3Ccircle cx='50' cy='90' r='2'/%3E%3Ccircle cx='100' cy='20' r='1.5'/%3E%3C/g%3E%3C/svg%3E")`,
    }}>
      <WAHeader />
      <div style={{ flex: 1, overflow: 'auto', padding: '8px 0', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ alignSelf: 'center', background: 'rgba(255,255,255,.92)', padding: '4px 10px', borderRadius: 8, fontSize: 12, color: '#54656F', fontFamily: '-apple-system', marginBottom: 6 }}>TODAY</div>
        {msgs}
      </div>
      <WACompose />
    </div>
  );
}

Object.assign(window, { WhatsAppChat });
