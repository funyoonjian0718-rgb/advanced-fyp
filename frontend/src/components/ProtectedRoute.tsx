import { Navigate, Outlet } from "react-router-dom";

function ProtectedRoute() {
    // Check whether a JWT access token exists in the browser.
    //
    // If the token exists, we assume the user has logged in.
    const token = localStorage.getItem("access_token");

    // If there is no token, the user is not authenticated.
    //
    // Navigate sends the user to the login page.
    // "replace" prevents the protected page from remaining
    // in the browser history.
    if (!token) {
        return <Navigate to="/login" replace />;
    }

    // If the token exists, allow the requested page to render.
    //
    // Outlet represents whichever child route is being accessed.
    return <Outlet />;
}

export default ProtectedRoute;