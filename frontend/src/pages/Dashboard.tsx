import { Link } from "react-router-dom";

const shortcuts = [
    { to: "/chat", icon: "✳", title: "Ask your dietary assistant", description: "Ask about Malaysian foods, nutrition values, or general dietary guidance." },
    { to: "/profile", icon: "◎", title: "Personalise your profile", description: "Keep your dietary preferences and known allergies up to date." },
    { to: "/history", icon: "◷", title: "Continue exploring", description: "Revisit previous questions and the answers saved in your history." },
];

function Dashboard() {
    return (
        <div className="dashboard-wrap">
            <section className="dashboard-hero">
                <div className="dashboard-hero-copy">
                    <span className="eyebrow">YOUR PERSONAL DIETARY SPACE</span>
                    <h1>Eat well.<br />Feel well.</h1>
                    <p>Explore nutrition information and dietary guidance with a little more clarity. Start with a question and see the available evidence behind the answer.</p>
                    <Link to="/chat" className="mt-6 inline-flex items-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold no-underline" style={{ background: "var(--accent)", color: "#102016" }}>
                        Ask a question <span aria-hidden="true">↗</span>
                    </Link>
                </div>
                <div className="hero-orbit" aria-hidden="true">♧</div>
            </section>

            <section>
                <div className="dashboard-section-head">
                    <div>
                        <h2>Make this space yours</h2>
                        <p>Choose where you would like to start.</p>
                    </div>
                </div>
                <div className="quick-grid">
                    {shortcuts.map((item) => (
                        <Link className="quick-card" to={item.to} key={item.to}>
                            <span className="quick-icon" aria-hidden="true">{item.icon}</span>
                            <strong>{item.title}</strong>
                            <span>{item.description}</span>
                        </Link>
                    ))}
                </div>
            </section>

            <aside className="dashboard-tip">
                <span className="dashboard-tip-icon" aria-hidden="true">✦</span>
                <div>
                    <strong>A note about dietary advice</strong>
                    <p>Nutrition values depend on food, ingredients and serving size. Check the cited sources and treat estimates as estimates. This tool is for information and does not replace professional medical advice.</p>
                </div>
            </aside>
        </div>
    );
}

export default Dashboard;
