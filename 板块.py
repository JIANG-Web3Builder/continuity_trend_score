import os
import pandas as pd
from pytdx.hq import TdxHq_API
import time
from datetime import datetime
import re

class SectorDataCollector:
    def __init__(self, max_workers=5):
        self.api = TdxHq_API()
        self.servers = [
            ('124.71.187.122', 7709),
        ]
        self.base_path = 'sector_data'
        self.start_date = '2026-05-04'
        self.end_date = '2026-06-01'
        self.markets = [1, 0, 2]
        
        self.sector_files = [
            ('行业板块.txt', '行业板块'),
            ('地区板块.txt', '地区板块'),
            ('风格板块.txt', '风格板块'),
            ('概念板块.txt', '概念板块'), 
            ('指数板块.txt', '指数板块')
        ]
        
        self.connected_server = None

    def connect_api(self):
        for server_ip, server_port in self.servers:
            try:
                print(f"尝试连接服务器: {server_ip}:{server_port}")
                if self.api.connect(server_ip, server_port, time_out=5):
                    print(f"连接成功: {server_ip}:{server_port}")
                    self.connected_server = (server_ip, server_port)
                    return True
            except Exception as e:
                print(f"连接 {server_ip}:{server_port} 失败: {e}")
                continue
        print("所有服务器连接失败")
        return False

    def ensure_connection(self):
        if self.connected_server is None:
            return self.connect_api()
        try:
            test_data = self.api.get_index_bars(9, 1, '000001', 0, 1)
            if test_data is not None:
                return True
        except:
            pass
        print("检测到连接已断开，尝试重新连接...")
        return self.reconnect_api()

    def reconnect_api(self):
        print("尝试重新连接API...")
        if self.connected_server:
            server_ip, server_port = self.connected_server
            try:
                self.api.disconnect()
                if self.api.connect(server_ip, server_port, time_out=5):
                    print(f"重新连接成功: {server_ip}:{server_port}")
                    return True
            except Exception as e:
                print(f"重新连接失败: {e}")
        return self.connect_api()

    def parse_sector_file(self, file_path):
        """健壮地解析板块文件，自动尝试多种编码"""
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            return []
        
        # 尝试多种编码
        encodings_to_try = ['gb18030', 'utf-8', 'gbk', 'gb2312', 'latin-1']
        sector_dict = {}
        
        for encoding in encodings_to_try:
            try:
                print(f"尝试使用编码 {encoding} 读取 {file_path}...")
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    lines = f.readlines()
                # 成功读取，开始解析
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        sector_code = parts[0].strip()
                        sector_name = parts[1].strip()
                        sector_code = re.sub(r'\s+', '', sector_code)
                        if sector_code and sector_code not in sector_dict:
                            sector_dict[sector_code] = sector_name
                    else:
                        # 不打印每一行错误，避免过多输出
                        pass
                print(f"成功使用编码 {encoding} 解析，共 {len(sector_dict)} 个板块")
                break  # 解析成功，跳出编码循环
            except UnicodeDecodeError as e:
                print(f"编码 {encoding} 失败: {e}")
                continue
            except Exception as e:
                print(f"其他错误 (编码 {encoding}): {e}")
                continue
        
        if not sector_dict:
            # 最后尝试二进制模式按行解码
            print("所有编码尝试失败，尝试逐行强制解码...")
            with open(file_path, 'rb') as f:
                for line_num, raw_line in enumerate(f, 1):
                    try:
                        line = raw_line.decode('gb18030', errors='ignore').strip()
                    except:
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        sector_code = parts[0].strip()
                        sector_name = parts[1].strip()
                        sector_code = re.sub(r'\s+', '', sector_code)
                        if sector_code and sector_code not in sector_dict:
                            sector_dict[sector_code] = sector_name
        
        sectors = [{'code': code, 'name': name} for code, name in sector_dict.items()]
        print(f"从 {file_path} 最终解析到 {len(sectors)} 个板块")
        if len(sectors) <= 10:
            for s in sectors:
                print(f"  - {s['code']}: {s['name']}")
        return sectors

    def get_sector_kline(self, sector_info):
        sector_code = sector_info['code']
        sector_name = sector_info['name']
        
        for attempt in range(3):
            for market in self.markets:
                try:
                    print(f"  获取 {sector_name}({sector_code}) 市场{market}，尝试 {attempt+1}/3")
                    data = self.api.get_index_bars(
                        category=9,
                        market=market,
                        code=sector_code,
                        start=0,
                        count=800
                    )
                    
                    if not data:
                        data = self.api.get_security_bars(
                            category=9,
                            market=market,
                            code=sector_code,
                            start=0,
                            count=800
                        )
                    
                    if not data:
                        continue
                    
                    df = self.api.to_df(data)
                    if df.empty:
                        continue
                    
                    date_col = None
                    for col in ['datetime', 'trade_date', 'date']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if not date_col:
                        continue
                    
                    try:
                        df['date'] = pd.to_datetime(df[date_col])
                    except:
                        try:
                            df['date'] = pd.to_datetime(df[date_col], format='%Y%m%d')
                        except:
                            continue
                    
                    rename_map = {}
                    if 'open' in df.columns:
                        rename_map['open'] = 'open'
                    if 'high' in df.columns:
                        rename_map['high'] = 'high'
                    if 'low' in df.columns:
                        rename_map['low'] = 'low'
                    if 'close' in df.columns:
                        rename_map['close'] = 'close'
                    if 'vol' in df.columns:
                        rename_map['vol'] = 'volume'
                    if 'amount' in df.columns:
                        rename_map['amount'] = 'amount'
                    if rename_map:
                        df = df.rename(columns=rename_map)
                    
                    required = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if any(col not in df.columns for col in required):
                        continue
                    
                    start_dt = pd.to_datetime(self.start_date)
                    end_dt = pd.to_datetime(self.end_date)
                    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    if df.empty:
                        continue
                    
                    df = df.sort_values('date').reset_index(drop=True)
                    final_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if 'amount' in df.columns:
                        final_cols.append('amount')
                    df = df[final_cols]
                    df['date'] = df['date'].dt.strftime('%Y/%m/%d')
                    
                    print(f"  成功获取 {sector_code} (市场{market})，共 {len(df)} 条")
                    return df
                    
                except Exception as e:
                    continue
            
            if attempt < 2:
                time.sleep(1)
                self.ensure_connection()
        
        print(f"  最终失败: {sector_code}")
        return None

    def save_sector_csv(self, sector_info, df, folder_name):
        sector_code = sector_info['code']
        sector_name = sector_info['name']
        clean_name = re.sub(r'[\\/*?:"<>|]', '', sector_name)
        filename = f"{sector_code}_{clean_name}.csv"
        sub_path = os.path.join(self.base_path, folder_name)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        filepath = os.path.join(sub_path, filename)
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  已保存: {filepath}")
            return True
        except Exception as e:
            print(f"  保存CSV失败: {e}")
            return False

    def run(self):
        print("开始获取板块指数数据...")
        print(f"时间范围: {self.start_date} 到 {self.end_date}")
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        
        total_success = 0
        total_sectors = 0
        
        for file_name, folder_name in self.sector_files:
            print(f"\n=== 处理文件: {file_name} ===")
            
            if not self.ensure_connection():
                print(f"连接失败，跳过文件 {file_name}")
                continue
            
            sectors = self.parse_sector_file(file_name)
            if not sectors:
                print(f"文件 {file_name} 无有效板块数据，跳过")
                continue
            
            total_sectors += len(sectors)
            success_count = 0
            
            for idx, sector in enumerate(sectors):
                print(f"进度: {idx+1}/{len(sectors)}")
                df = self.get_sector_kline(sector)
                if df is not None and self.save_sector_csv(sector, df, folder_name):
                    success_count += 1
                time.sleep(0.5)
                
                if (idx + 1) % 10 == 0:
                    self.ensure_connection()
            
            print(f"文件 {file_name} 完成: 成功 {success_count}/{len(sectors)} 个板块")
            total_success += success_count
        
        print(f"\n全部完成！成功获取 {total_success}/{total_sectors} 个板块的数据")
        return total_success > 0


def test_api_connection():
    api = TdxHq_API()
    if api.connect('119.29.201.30', 7709):
        print("API连接成功")
        try:
            data = api.get_index_bars(9, 1, '000001', 0, 10)
            if data and len(data) > 0:
                df = api.to_df(data)
                print("成功获取上证指数数据:")
                print(df.head())
            else:
                print("上证指数数据获取失败")
        except Exception as e:
            print(f"获取数据时出错: {e}")
        api.disconnect()
    else:
        print("API连接失败")

def main():
    print("=== 开始API诊断 ===")
    test_api_connection()
    print("=== API诊断完成 ===\n")
    
    collector = SectorDataCollector()
    collector.run()

if __name__ == "__main__":
    main()