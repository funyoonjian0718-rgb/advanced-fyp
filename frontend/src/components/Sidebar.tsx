import { NavLink, useNavigate } from "react-router-dom";

import { logout } from "../services/auth";

function Sidebar() {
    // Allows us to redirect after logout.
    const navigate = useNavigate();

    // This function handles the logout process.
    function handleLogout() {
        // Remove the JWT and stored user information.
        logout();

        // Send the user back to the login page.
        navigate("/login", { replace: true });
    }

    return (
        <aside className="w-64 min-h-screen bg-gray-900 text-white p-4">

            {/* Application name */}
            <h1 className="text-xl font-bold mb-8">
                Advanced FYP
            </h1>

            {/* Navigation links */}
            <nav className="space-y-2">

                <NavLink
                    to="/dashboard"
                    className="block p-2 rounded hover:bg-gray-800"
                >
                    Dashboard
                </NavLink>

                <NavLink
                    to="/chat"
                    className="block p-2 rounded hover:bg-gray-800"
                >
                    Dietary Assistant
                </NavLink>

                <NavLink
                    to="/history"
                    className="block p-2 rounded hover:bg-gray-800"
                >
                    History
                </NavLink>

                <NavLink
                    to="/profile"
                    className="block p-2 rounded hover:bg-gray-800"
                >
                    Profile
                </NavLink>

            </nav>

            {/* 
                Logout button.

                mt-auto pushes this button toward the
                bottom when the sidebar uses flex layout.
            */}
            <button
                onClick={handleLogout}
                className="mt-8 w-full text-left p-2 rounded hover:bg-gray-800"
            >
                Logout
            </button>

        </aside>
    );
}

export default Sidebar;