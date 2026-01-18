import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Sample mock data 
const MOCK_DATA = Array.from({ length: 10 }, (_, i) => ({
    Unit: i + 1,
    Prediction: Math.floor(Math.random() * 125),
    Health_Percentage: Math.floor(Math.random() * 100),
}));

function Dashboard() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedUnit, setSelectedUnit] = useState(null);

    useEffect(() => {
        // Fetch the prediction result with cache busting
        fetch(`/final_submission.csv?t=${Date.now()}`)
            .then(response => {
                if (!response.ok) throw new Error("File not found");
                return response.text();
            })
            .then(csvText => {
                const rows = csvText.trim().split('\n').slice(1); // Skip header
                const parsed = rows.map(row => {
                    const [u, pred, health] = row.split(',');
                    return {
                        Unit: parseInt(u),
                        Prediction: parseFloat(pred).toFixed(1),
                        Health_Percentage: parseFloat(health).toFixed(1),
                    };
                });

                // Sort by Unit ID to ensure it starts from 1
                parsed.sort((a, b) => a.Unit - b.Unit);

                setData(parsed);
                if (parsed.length > 0) setSelectedUnit(parsed[0]);

                setLoading(false);
            })
            .catch(err => {
                console.warn("Could not load CSV, using mock data", err);
                setData(MOCK_DATA);
                setSelectedUnit(MOCK_DATA[0]);
                setLoading(false);
            });
    }, []);

    const getHealthColor = (pct) => {
        if (pct > 70) return '#10b981'; // Green
        if (pct > 30) return '#f59e0b'; // Yellow
        return '#ef4444'; // Red
    };

    if (loading) return <div className="glass card">Loading Engine Data...</div>;

    return (
        <div className="dashboard-container">
            <h1>Engine Health Dashboard</h1>

            {selectedUnit && (
                <div className="dashboard-grid">
                    {/* Main Health Card */}
                    <div className="glass card main-stat" style={{ borderTop: `4px solid ${getHealthColor(selectedUnit.Health_Percentage)}` }}>
                        <h2>Unit #{selectedUnit.Unit} Status</h2>
                        <div
                            className="health-circle"
                            style={{
                                '--percentage': `${selectedUnit.Health_Percentage}%`,
                                '--curr-color': getHealthColor(selectedUnit.Health_Percentage)
                            }}
                        >
                            <div className="health-value" style={{ color: getHealthColor(selectedUnit.Health_Percentage) }}>
                                {Math.round(selectedUnit.Health_Percentage)}%
                            </div>
                        </div>
                        <div className="stat-label">Health Score</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '5px' }}>
                            (0% = Failure, 100% = New)
                        </div>
                        <div style={{ marginTop: '0.5rem', fontWeight: 'bold', color: getHealthColor(selectedUnit.Health_Percentage) }}>
                            {selectedUnit.Health_Percentage > 70 ? 'Good Condition' :
                                selectedUnit.Health_Percentage > 30 ? 'Warning State' : 'Critical Failure'}
                        </div>
                    </div>

                    {/* Cycle Prediction Card */}
                    <div className="glass card">
                        <h2>Est. Cycles</h2>
                        <div className="vul-highlight">
                            {Math.round(selectedUnit.Prediction)}
                        </div>
                        <div className="stat-label">Remaining Useful Life (Est)</div>
                    </div>

                    {/* Action Card */}
                    <div className="glass card" style={{ borderLeft: `4px solid ${getHealthColor(selectedUnit.Health_Percentage)}` }}>
                        <h2>Recommendation</h2>
                        <div style={{ textAlign: 'center', margin: 'auto' }}>
                            <p style={{ fontSize: '1.2rem', fontWeight: '600', color: getHealthColor(selectedUnit.Health_Percentage) }}>
                                {selectedUnit.Health_Percentage > 70 ? "No Action Needed" :
                                    selectedUnit.Health_Percentage > 30 ? "Schedule Inspection" :
                                        "IMMEDIATE OVERHAUL"}
                            </p>
                            <p style={{ opacity: 0.8, fontSize: '0.9rem' }}>
                                {selectedUnit.Health_Percentage > 70 ? "Engine is operating within normal parameters." :
                                    selectedUnit.Health_Percentage > 30 ? "Vibration levels may be increasing. Check logs." :
                                        "Safety threshold approached. Ground aircraft."}
                            </p>
                        </div>
                        <button
                            className="btn"
                            style={{ background: getHealthColor(selectedUnit.Health_Percentage) }}
                            onClick={() => alert("Maintenance Ticket Created")}
                        >
                            Create Ticket
                        </button>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="status-dot" style={{ background: '#10b981' }} /> Good (&gt;70%)
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="status-dot" style={{ background: '#f59e0b' }} /> Warning (30-70%)
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="status-dot" style={{ background: '#ef4444' }} /> Critical (&lt;30%)
                </div>
            </div>

            {/* Visualization Charts */}
            <div className="glass card" style={{ marginTop: '2rem', padding: '2rem' }}>
                <h2>Prediction Overview</h2>
                <div style={{ height: 300, width: '100%', marginTop: '1rem' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}
                            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="Unit" stroke="#94a3b8" />
                            <YAxis stroke="#94a3b8" />
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
                                itemStyle={{ color: '#e2e8f0' }}
                            />
                            <Legend />
                            <Area type="monotone" name="Cycles" dataKey="Prediction" stroke="#8884d8" fillOpacity={1} fill="url(#colorPred)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* List of Units */}
            <div className="glass card data-table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Unit ID</th>
                            <th>Status</th>
                            <th>Prediction</th>
                            <th>Health Score</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map(unit => (
                            <tr
                                key={unit.Unit}
                                onClick={() => setSelectedUnit(unit)}
                                style={{
                                    background: selectedUnit?.Unit === unit.Unit ? 'rgba(255,255,255,0.1)' : 'transparent',
                                    cursor: 'pointer'
                                }}
                            >
                                <td>#{unit.Unit}</td>
                                <td style={{ color: getHealthColor(unit.Health_Percentage), fontWeight: 'bold' }}>
                                    <span
                                        className="status-dot"
                                        style={{ background: getHealthColor(unit.Health_Percentage) }}
                                    />
                                    {unit.Health_Percentage > 70 ? 'Good' : unit.Health_Percentage > 30 ? 'Warning' : 'Critical'}
                                </td>
                                <td>{Math.round(unit.Prediction)}</td>
                                <td style={{ color: getHealthColor(unit.Health_Percentage), fontWeight: 'bold' }}>
                                    {Math.round(unit.Health_Percentage)}%
                                </td>
                                <td><button size="sm" style={{ padding: '4px 8px', borderRadius: '4px', border: 'none', background: 'rgba(255,255,255,0.1)', color: 'white' }}>View</button></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default Dashboard;
