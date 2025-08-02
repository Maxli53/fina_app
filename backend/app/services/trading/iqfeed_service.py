"""IQFeed data provider service"""

import socket
import asyncio
import logging
import os
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from app.models.trading import (
    IQFeedConfig, DataProviderConnection, ConnectionStatus,
    IQFeedMarketData, IQFeedHistoricalRequest, IQFeedDataType,
    MarketDataSnapshot, DataProviderType
)

logger = logging.getLogger(__name__)


class IQFeedService:
    """Service for IQFeed real-time and historical data"""
    
    def __init__(self):
        self.config: Optional[IQFeedConfig] = None
        self.connection_status = DataProviderConnection(
            provider_type=DataProviderType.IQFEED,
            status=ConnectionStatus.DISCONNECTED
        )
        
        # Socket connections
        self.level1_socket: Optional[socket.socket] = None
        self.level2_socket: Optional[socket.socket] = None
        self.admin_socket: Optional[socket.socket] = None
        self.historical_socket: Optional[socket.socket] = None
        
        # Data handlers
        self.market_data_handlers: Dict[str, List[Callable]] = {}
        self.watched_symbols: set = set()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Data buffers
        self.latest_quotes: Dict[str, IQFeedMarketData] = {}
        
    async def initialize(self, config: Optional[IQFeedConfig] = None) -> bool:
        """Initialize IQFeed connection"""
        try:
            # Load config from environment if not provided
            if config is None:
                config = IQFeedConfig(
                    login=os.getenv("IQFEED_LOGIN", ""),
                    password=os.getenv("IQFEED_PASSWORD", ""),
                    product_id=os.getenv("IQFEED_PRODUCT_ID", "FINANCIAL_TIME_SERIES_PLATFORM"),
                    product_version=os.getenv("IQFEED_PRODUCT_VERSION", "1.0")
                )
            
            self.config = config
            
            # Connect to admin port first for authentication
            success = await self._connect_admin()
            if not success:
                return False
            
            # Connect to data services
            await self._connect_level1()
            await self._connect_historical()
            
            self.connection_status.status = ConnectionStatus.CONNECTED
            self.connection_status.connected_at = datetime.utcnow()
            self.running = True
            
            # Start data processing loops
            asyncio.create_task(self._process_level1_data())
            asyncio.create_task(self._heartbeat_loop())
            
            logger.info("IQFeed service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IQFeed: {str(e)}")
            self.connection_status.status = ConnectionStatus.ERROR
            self.connection_status.error_message = str(e)
            return False
    
    async def _connect_admin(self) -> bool:
        """Connect to IQFeed admin port for authentication"""
        try:
            self.admin_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.admin_socket.settimeout(self.config.connection_timeout)
            self.admin_socket.connect((self.config.host, self.config.admin_port))
            
            # Send authentication
            auth_msg = f"S,CONNECT CLIENT,{self.config.product_id},{self.config.product_version}\r\n"
            self.admin_socket.send(auth_msg.encode())
            
            # Send login credentials
            login_msg = f"S,LOGIN,{self.config.login},{self.config.password}\r\n"
            self.admin_socket.send(login_msg.encode())
            
            # Wait for confirmation
            response = self.admin_socket.recv(1024).decode()
            if "CONNECTED" in response:
                self.connection_status.admin_connected = True
                logger.info("IQFeed admin connection established")
                return True
            else:
                logger.error(f"IQFeed admin connection failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Admin connection error: {str(e)}")
            return False
    
    async def _connect_level1(self) -> bool:
        """Connect to Level 1 data feed"""
        try:
            self.level1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.level1_socket.settimeout(1.0)  # Non-blocking for async
            self.level1_socket.connect((self.config.host, self.config.level1_port))
            
            # Set protocol
            self.level1_socket.send(b"S,SET PROTOCOL,6.2\r\n")
            
            self.connection_status.level1_connected = True
            logger.info("IQFeed Level 1 connection established")
            return True
            
        except Exception as e:
            logger.error(f"Level 1 connection error: {str(e)}")
            return False
    
    async def _connect_historical(self) -> bool:
        """Connect to historical data feed"""
        try:
            self.historical_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.historical_socket.settimeout(self.config.connection_timeout)
            self.historical_socket.connect((self.config.host, self.config.historical_port))
            
            # Set protocol
            self.historical_socket.send(b"S,SET PROTOCOL,6.2\r\n")
            
            self.connection_status.historical_connected = True
            logger.info("IQFeed Historical connection established")
            return True
            
        except Exception as e:
            logger.error(f"Historical connection error: {str(e)}")
            return False
    
    async def watch_symbol(self, symbol: str, handler: Optional[Callable] = None) -> bool:
        """Subscribe to real-time data for a symbol"""
        try:
            if self.level1_socket is None:
                logger.error("Level 1 connection not established")
                return False
            
            # Add to watched symbols
            self.watched_symbols.add(symbol)
            
            # Add handler if provided
            if handler:
                if symbol not in self.market_data_handlers:
                    self.market_data_handlers[symbol] = []
                self.market_data_handlers[symbol].append(handler)
            
            # Send watch command
            watch_cmd = f"w{symbol}\r\n"
            self.level1_socket.send(watch_cmd.encode())
            
            logger.info(f"Watching symbol: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to watch symbol {symbol}: {str(e)}")
            return False
    
    async def unwatch_symbol(self, symbol: str) -> bool:
        """Unsubscribe from real-time data for a symbol"""
        try:
            if self.level1_socket is None:
                return False
            
            # Remove from watched symbols
            self.watched_symbols.discard(symbol)
            
            # Send unwatch command
            unwatch_cmd = f"r{symbol}\r\n"
            self.level1_socket.send(unwatch_cmd.encode())
            
            # Clear handlers
            if symbol in self.market_data_handlers:
                del self.market_data_handlers[symbol]
            
            logger.info(f"Unwatched symbol: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unwatch symbol {symbol}: {str(e)}")
            return False
    
    async def _process_level1_data(self):
        """Process incoming Level 1 data"""
        buffer = ""
        
        while self.running and self.level1_socket:
            try:
                # Non-blocking receive
                self.level1_socket.setblocking(False)
                try:
                    data = self.level1_socket.recv(4096).decode()
                    if not data:
                        await asyncio.sleep(0.01)
                        continue
                except socket.error:
                    await asyncio.sleep(0.01)
                    continue
                
                buffer += data
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith('Q,'):  # Quote update
                        await self._process_quote(line)
                    elif line.startswith('F,'):  # Fundamental data
                        await self._process_fundamental(line)
                    elif line.startswith('T,'):  # Trade update
                        await self._process_trade(line)
                    elif line.startswith('S,'):  # System message
                        logger.info(f"IQFeed system message: {line}")
                        
            except Exception as e:
                logger.error(f"Error processing Level 1 data: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_quote(self, data: str):
        """Process quote update"""
        try:
            fields = data.split(',')
            if len(fields) < 20:
                return
            
            symbol = fields[1]
            
            # Parse quote data
            quote = IQFeedMarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                last=float(fields[3]) if fields[3] else 0,
                last_size=int(fields[7]) if fields[7] else 0,
                total_volume=int(fields[6]) if fields[6] else 0,
                bid=float(fields[10]) if fields[10] else 0,
                bid_size=int(fields[12]) if fields[12] else 0,
                ask=float(fields[11]) if fields[11] else 0,
                ask_size=int(fields[13]) if fields[13] else 0,
                open=float(fields[18]) if fields[18] else 0,
                high=float(fields[8]) if fields[8] else 0,
                low=float(fields[9]) if fields[9] else 0,
                close=float(fields[19]) if fields[19] else 0,
                prev_close=float(fields[20]) if fields[20] else 0,
                net_change=float(fields[4]) if fields[4] else 0,
                percent_change=float(fields[14]) if fields[14] else 0,
                vwap=float(fields[17]) if fields[17] else 0,
                tick_direction='up' if float(fields[4] or 0) > 0 else 'down',
                exchange=fields[15] if len(fields) > 15 else '',
                market_open=True
            )
            
            # Update latest quote
            self.latest_quotes[symbol] = quote
            
            # Update metrics
            self.connection_status.messages_received += 1
            self.connection_status.data_points_received += 1
            
            # Call handlers
            if symbol in self.market_data_handlers:
                for handler in self.market_data_handlers[symbol]:
                    try:
                        await handler(quote)
                    except Exception as e:
                        logger.error(f"Handler error for {symbol}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing quote: {str(e)}")
    
    async def _process_fundamental(self, data: str):
        """Process fundamental data update"""
        # Implement fundamental data processing if needed
        pass
    
    async def _process_trade(self, data: str):
        """Process trade update"""
        # Implement trade data processing if needed
        pass
    
    async def get_historical_data(
        self, 
        request: IQFeedHistoricalRequest
    ) -> Optional[pd.DataFrame]:
        """Get historical data from IQFeed"""
        try:
            if self.historical_socket is None:
                logger.error("Historical connection not established")
                return None
            
            # Build request command based on data type
            if request.data_type == IQFeedDataType.HISTORICAL_DAILY:
                cmd = self._build_daily_request(request)
            elif request.data_type == IQFeedDataType.HISTORICAL_INTERVAL:
                cmd = self._build_interval_request(request)
            elif request.data_type == IQFeedDataType.HISTORICAL_TICK:
                cmd = self._build_tick_request(request)
            else:
                logger.error(f"Unsupported data type: {request.data_type}")
                return None
            
            # Send request
            self.historical_socket.send(cmd.encode())
            
            # Receive data
            data_buffer = ""
            complete = False
            
            while not complete:
                chunk = self.historical_socket.recv(65536).decode()
                data_buffer += chunk
                
                # Check for end marker
                if "!ENDMSG!" in data_buffer:
                    complete = True
                elif "!NO_DATA!" in data_buffer:
                    logger.warning(f"No data available for {request.symbol}")
                    return None
                elif "!ERROR!" in data_buffer:
                    logger.error(f"IQFeed error: {data_buffer}")
                    return None
            
            # Parse data
            df = self._parse_historical_data(data_buffer, request.data_type)
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def _build_daily_request(self, request: IQFeedHistoricalRequest) -> str:
        """Build daily data request command"""
        if request.num_days:
            return f"HDX,{request.symbol},{request.num_days},0,,,,BEGINDATE,ENDDATE,\r\n"
        elif request.start_date and request.end_date:
            start = request.start_date.strftime("%Y%m%d")
            end = request.end_date.strftime("%Y%m%d")
            return f"HDT,{request.symbol},{start} 000000,{end} 235959,,,,BEGINDATE,ENDDATE,\r\n"
        else:
            return f"HDX,{request.symbol},100,0,,,,BEGINDATE,ENDDATE,\r\n"
    
    def _build_interval_request(self, request: IQFeedHistoricalRequest) -> str:
        """Build interval data request command"""
        interval = request.interval_seconds or 60
        
        if request.num_days:
            return f"HIX,{request.symbol},{interval},{request.num_days},,,BEGINDATE,ENDDATE,\r\n"
        elif request.start_date and request.end_date:
            start = request.start_date.strftime("%Y%m%d %H%M%S")
            end = request.end_date.strftime("%Y%m%d %H%M%S")
            return f"HIT,{request.symbol},{interval},{start},{end},,BEGINDATE,ENDDATE,\r\n"
        else:
            return f"HIX,{request.symbol},{interval},10,,,BEGINDATE,ENDDATE,\r\n"
    
    def _build_tick_request(self, request: IQFeedHistoricalRequest) -> str:
        """Build tick data request command"""
        if request.num_days:
            return f"HTX,{request.symbol},{request.num_days},{request.max_datapoints or 10000},,BEGINDATE,ENDDATE,\r\n"
        elif request.start_date and request.end_date:
            start = request.start_date.strftime("%Y%m%d %H%M%S")
            end = request.end_date.strftime("%Y%m%d %H%M%S")
            return f"HTT,{request.symbol},{start},{end},{request.max_datapoints or 10000},,BEGINDATE,ENDDATE,\r\n"
        else:
            return f"HTX,{request.symbol},1,10000,,BEGINDATE,ENDDATE,\r\n"
    
    def _parse_historical_data(self, data: str, data_type: IQFeedDataType) -> pd.DataFrame:
        """Parse historical data response"""
        lines = data.strip().split('\n')
        
        # Remove protocol and end messages
        data_lines = [line for line in lines if not line.startswith('!') and line.strip()]
        
        if not data_lines:
            return pd.DataFrame()
        
        # Parse based on data type
        if data_type == IQFeedDataType.HISTORICAL_DAILY:
            columns = ['datetime', 'high', 'low', 'open', 'close', 'volume', 'open_interest']
            df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
            
            # Convert types
            df['datetime'] = pd.to_datetime(df['datetime'])
            for col in ['high', 'low', 'open', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')
            
        elif data_type == IQFeedDataType.HISTORICAL_INTERVAL:
            columns = ['datetime', 'high', 'low', 'open', 'close', 'volume', 'open_interest']
            df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
            
            # Convert types
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
            for col in ['high', 'low', 'open', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')
            
        elif data_type == IQFeedDataType.HISTORICAL_TICK:
            columns = ['datetime', 'last', 'last_size', 'total_volume', 'bid', 'ask', 'tick_id']
            df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
            
            # Convert types
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
            for col in ['last', 'bid', 'ask']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            for col in ['last_size', 'total_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
        else:
            return pd.DataFrame()
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    async def get_latest_quote(self, symbol: str) -> Optional[MarketDataSnapshot]:
        """Get latest quote for a symbol"""
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            
            # Convert to unified format
            return MarketDataSnapshot(
                symbol=symbol,
                timestamp=quote.timestamp,
                source=DataProviderType.IQFEED,
                bid=quote.bid,
                ask=quote.ask,
                last=quote.last,
                mid=(quote.bid + quote.ask) / 2,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                last_size=quote.last_size,
                volume=quote.total_volume,
                vwap=quote.vwap,
                change=quote.net_change,
                change_percent=quote.percent_change,
                open=quote.open,
                high=quote.high,
                low=quote.low,
                close=quote.close,
                prev_close=quote.prev_close,
                halted=False,
                trading_status="normal" if quote.market_open else "closed"
            )
        
        return None
    
    async def _heartbeat_loop(self):
        """Send heartbeat to maintain connection"""
        while self.running:
            try:
                if self.admin_socket:
                    self.admin_socket.send(b"S,HEARTBEAT\r\n")
                    self.connection_status.last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(5)
    
    async def disconnect(self):
        """Disconnect from IQFeed"""
        self.running = False
        
        # Close all sockets
        for sock in [self.level1_socket, self.level2_socket, 
                     self.admin_socket, self.historical_socket]:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        self.connection_status.status = ConnectionStatus.DISCONNECTED
        self.connection_status.connected_at = None
        
        logger.info("IQFeed service disconnected")
    
    def get_connection_status(self) -> DataProviderConnection:
        """Get current connection status"""
        return self.connection_status