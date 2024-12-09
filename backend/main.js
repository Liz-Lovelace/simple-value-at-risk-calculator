import yahooFinance from 'yahoo-finance2';

const data = await yahooFinance.quote('AAPL');
console.log(data);