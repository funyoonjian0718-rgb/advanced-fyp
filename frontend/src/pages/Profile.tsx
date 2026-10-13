import { useEffect, useState, type FormEvent } from "react";
import { isAxiosError } from "axios";
import { useLocation, useNavigate } from "react-router-dom";

import {
    getProfile,
    updateProfile,
    type UserProfile,
    type UserProfileUpdate,
} from "../services/profile";

interface ProfileForm {
    full_name: string;
    age: string;
    gender: string;
    height_cm: string;
    weight_kg: string;
    activity_level: string;
    dietary_preference: string;
    allergies: string;
}

const emptyForm: ProfileForm = {
    full_name: "",
    age: "",
    gender: "",
    height_cm: "",
    weight_kg: "",
    activity_level: "",
    dietary_preference: "",
    allergies: "",
};

function toFormValue(value: number | string | null): string {
    return value === null ? "" : String(value);
}

function toOptionalNumber(value: string): number | null {
    if (value.trim() === "") {
        return null;
    }

    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function getErrorMessage(error: unknown): string {
    if (isAxiosError<{ detail?: unknown }>(error)) {
        const detail = error.response?.data?.detail;
        if (typeof detail === "string") {
            return detail;
        }
    }

    return "Something went wrong. Please try again.";
}

function profileToForm(profile: UserProfile): ProfileForm {
    return {
        full_name: profile.full_name ?? "",
        age: toFormValue(profile.age),
        gender: profile.gender ?? "",
        height_cm: toFormValue(profile.height_cm),
        weight_kg: toFormValue(profile.weight_kg),
        activity_level: profile.activity_level ?? "",
        dietary_preference: profile.dietary_preference ?? "",
        allergies: profile.allergies ?? "",
    };
}

function Profile() {
    const location = useLocation();
    const navigate = useNavigate();
    const onboarding = Boolean((location.state as { onboarding?: boolean } | null)?.onboarding);
    const [form, setForm] = useState<ProfileForm>(emptyForm);
    const [error, setError] = useState("");
    const [saved, setSaved] = useState(false);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        async function loadProfile() {
            try {
                const profile = await getProfile();
                setForm(profileToForm(profile));
            } catch (loadError: unknown) {
                if (
                    isAxiosError(loadError) &&
                    loadError.response?.status === 404
                ) {
                    setForm(emptyForm);
                } else {
                    setError(getErrorMessage(loadError));
                }
            } finally {
                setLoading(false);
            }
        }

        void loadProfile();
    }, []);

    function updateField(field: keyof ProfileForm, value: string) {
        setForm((current) => ({ ...current, [field]: value }));
        setSaved(false);
    }

    async function handleSubmit(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();
        setError("");
        setSaved(false);

        if (onboarding) {
            if (!form.full_name.trim()) {
                setError("Please enter your full name to continue.");
                return;
            }
            if (!form.age.trim() || Number(form.age) < 1 || Number(form.age) > 120) {
                setError("Please enter a valid age between 1 and 120.");
                return;
            }
            if (!form.dietary_preference.trim()) {
                setError("Please select or enter your dietary preference. If you have none, enter No preference.");
                return;
            }
            if (!form.allergies.trim()) {
                setError('Please list your food allergies, or enter "None" if you have none.');
                return;
            }
        }

        setSaving(true);

        const data: UserProfileUpdate = {
            full_name: form.full_name.trim() || null,
            age: toOptionalNumber(form.age),
            gender: form.gender || null,
            height_cm: toOptionalNumber(form.height_cm),
            weight_kg: toOptionalNumber(form.weight_kg),
            activity_level: form.activity_level || null,
            dietary_preference: form.dietary_preference.trim() || null,
            allergies: form.allergies.trim() || null,
        };

        try {
            const profile = await updateProfile(data);
            setForm(profileToForm(profile));
            setSaved(true);
            if (onboarding) {
                navigate("/dashboard", { replace: true, state: { profileCompleted: true } });
            }
        } catch (saveError: unknown) {
            setError(getErrorMessage(saveError));
        } finally {
            setSaving(false);
        }
    }

    const inputClassName = "w-full rounded border border-gray-300 px-3 py-2";
    const labelClassName = "mb-1 block text-sm font-medium text-gray-700";

    return (
        <section className="mx-auto max-w-3xl p-6">
            <h1 className="mb-2 text-2xl font-bold">{onboarding ? "Set up your dietary profile" : "Profile"}</h1>
            <p className="mb-6 text-gray-600">
                {onboarding
                    ? "Before you start, tell us a little about yourself so the assistant can take your preferences and reported allergies into account. Fields marked * are required. You can update these details later."
                    : "Update your details to personalize dietary recommendations."}
            </p>

            {error && (
                <p role="alert" className="mb-4 rounded bg-red-100 p-3 text-red-700">
                    {error}
                </p>
            )}

            {saved && (
                <p role="status" className="mb-4 rounded bg-green-100 p-3 text-green-700">
                    Profile saved.
                </p>
            )}

            {loading ? (
                <p role="status">Loading profile...</p>
            ) : (
                <form
                    onSubmit={handleSubmit}
                    className="grid grid-cols-1 gap-5 rounded-lg bg-white p-6 shadow sm:grid-cols-2"
                >
                    <div className="sm:col-span-2">
                        <label htmlFor="full_name" className={labelClassName}>
                            Full name{onboarding ? " *" : ""}
                        </label>
                        <input
                            id="full_name"
                            name="full_name"
                            type="text"
                            maxLength={150}
                            value={form.full_name}
                            onChange={(event) =>
                                updateField("full_name", event.target.value)
                            }
                            className={inputClassName}
                        />
                    </div>

                    <div>
                        <label htmlFor="age" className={labelClassName}>
                            Age{onboarding ? " *" : ""}
                        </label>
                        <input
                            id="age"
                            name="age"
                            type="number"
                            min={1}
                            max={120}
                            step={1}
                            value={form.age}
                            onChange={(event) => updateField("age", event.target.value)}
                            className={inputClassName}
                        />
                    </div>

                    <div>
                        <label htmlFor="gender" className={labelClassName}>
                            Gender
                        </label>
                        <select
                            id="gender"
                            name="gender"
                            value={form.gender}
                            onChange={(event) =>
                                updateField("gender", event.target.value)
                            }
                            className={inputClassName}
                        >
                            <option value="">Prefer not to say</option>
                            <option value="female">Female</option>
                            <option value="male">Male</option>
                            <option value="non-binary">Non-binary</option>
                            <option value="other">Other</option>
                        </select>
                    </div>

                    <div>
                        <label htmlFor="height_cm" className={labelClassName}>
                            Height (cm)
                        </label>
                        <input
                            id="height_cm"
                            name="height_cm"
                            type="number"
                            min="0.1"
                            max={300}
                            step="any"
                            value={form.height_cm}
                            onChange={(event) =>
                                updateField("height_cm", event.target.value)
                            }
                            className={inputClassName}
                        />
                    </div>

                    <div>
                        <label htmlFor="weight_kg" className={labelClassName}>
                            Weight (kg)
                        </label>
                        <input
                            id="weight_kg"
                            name="weight_kg"
                            type="number"
                            min="0.1"
                            max={700}
                            step="any"
                            value={form.weight_kg}
                            onChange={(event) =>
                                updateField("weight_kg", event.target.value)
                            }
                            className={inputClassName}
                        />
                    </div>

                    <div>
                        <label htmlFor="activity_level" className={labelClassName}>
                            Activity level
                        </label>
                        <select
                            id="activity_level"
                            name="activity_level"
                            value={form.activity_level}
                            onChange={(event) =>
                                updateField("activity_level", event.target.value)
                            }
                            className={inputClassName}
                        >
                            <option value="">Select activity level</option>
                            <option value="sedentary">Sedentary</option>
                            <option value="light">Light</option>
                            <option value="moderate">Moderate</option>
                            <option value="active">Active</option>
                            <option value="very_active">Very active</option>
                        </select>
                    </div>

                    <div>
                        <label
                            htmlFor="dietary_preference"
                            className={labelClassName}
                        >
                            Dietary preference{onboarding ? " *" : ""}
                        </label>
                        <input
                            id="dietary_preference"
                            name="dietary_preference"
                            type="text"
                            maxLength={100}
                            list="dietary-preferences"
                            value={form.dietary_preference}
                            onChange={(event) =>
                                updateField(
                                    "dietary_preference",
                                    event.target.value
                                )
                            }
                            className={inputClassName}
                        />
                        <datalist id="dietary-preferences">
                            <option value="No preference" />
                            <option value="Vegetarian" />
                            <option value="Vegan" />
                            <option value="Halal" />
                            <option value="Kosher" />
                            <option value="Pescatarian" />
                        </datalist>
                    </div>

                    <div className="sm:col-span-2">
                        <label htmlFor="allergies" className={labelClassName}>
                            Allergies{onboarding ? " *" : ""}
                        </label>
                        <textarea
                            id="allergies"
                            name="allergies"
                            rows={3}
                            maxLength={2000}
                            value={form.allergies}
                            onChange={(event) =>
                                updateField("allergies", event.target.value)
                            }
                            className={inputClassName}
                            placeholder='List food allergies or intolerances, or enter "None"' 
                        />
                    </div>

                    <div className="sm:col-span-2">
                        <button
                            type="submit"
                            disabled={saving}
                            className="rounded bg-blue-600 px-5 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                        >
                            {saving ? "Saving..." : onboarding ? "Save and continue" : "Save profile"}
                        </button>
                    </div>
                </form>
            )}
        </section>
    );
}

export default Profile;
