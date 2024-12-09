import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import yahooFinance from 'yahoo-finance2';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

// Serve static files from public directory
app.use(express.static(path.join(__dirname, '../public')));

// Endpoint to get historical price data
app.get('/historicalPriceDataForTicker/:ticker', async (req, res) => {
    try {
        const ticker = req.params.ticker;
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 1); // Get 1 year of data

        const queryOptions = {
            period1: startDate,
            period2: endDate,
            interval: '1d'
        };

        const result = await yahooFinance.historical(ticker, queryOptions);
        const formattedData = result.map(item => ({
            date: item.date,
            price: item.close
        }));

        res.json(formattedData);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});