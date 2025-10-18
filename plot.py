import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io

st.set_page_config(page_title="Plasma-Jet Spectral Analysis", layout="wide")

st.title("Plasma-Jet Spectral Intensity & Current Waveform Dashboard")
st.markdown("*Advanced visualization with clustered bar layout and independent plot controls*")

# File upload section
st.sidebar.header("📁 Data Files")
st.sidebar.info("💡 Tip: Sine wave can be displayed with just waveform data (no peaks required)")

# Peak intensity files
square_peak_file = st.sidebar.file_uploader("Square Peak Data", type=['csv'], key='square_peak')
triangle_peak_file = st.sidebar.file_uploader("Triangle Peak Data", type=['csv'], key='triangle_peak')
sine_peak_file = st.sidebar.file_uploader("Sine Peak Data (Optional)", type=['csv'], key='sine_peak')

# Waveform files
square_wave_file = st.sidebar.file_uploader("Square Waveform Data", type=['csv'], key='square_wave')
triangle_wave_file = st.sidebar.file_uploader("Triangle Waveform Data", type=['csv'], key='triangle_wave')
sine_wave_file = st.sidebar.file_uploader("Sine Waveform Data", type=['csv'], key='sine_wave')

# Global settings
st.sidebar.header("⚙️ General Settings")
waveforms_to_plot = st.sidebar.multiselect(
    "Select waveforms to display",
    ["Sine", "Square", "Triangle"],
    default=["Sine", "Square", "Triangle"]
)

st.sidebar.info("""
**📊 Axis Behavior:**
- **Intensity (left)**: Values always positive (≥ 0)
- **Visual Invert**: Display bars below zero while keeping positive labels
- **Current (right)**: Can be positive/negative
- **Clustered Layout**: Bars automatically grouped within each time step
""")

# Font settings
st.sidebar.subheader("🔤 Font Settings")
font_family = st.sidebar.selectbox("Font family", 
    ["Times New Roman", "Arial", "Helvetica", "sans-serif", "serif"], 
    index=0)

# Global color settings
st.sidebar.subheader("🎨 Global Colors")
col1, col2 = st.sidebar.columns(2)
color_310 = col1.color_picker("310 nm", "#5DA5DA")
color_337 = col2.color_picker("337 nm", "#FAA43A")
color_696 = col1.color_picker("696 nm", "#60BD68")
color_current = col2.color_picker("Current", "#FF0000")

# Function to load and process data
@st.cache_data
def load_peak_data(file):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        return df
    return None

@st.cache_data
def load_waveform_data(file):
    if file is not None:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        if 'Time' in df.columns:
            df['Time_us'] = df['Time'] * 1e6
        return df
    return None

# Load all data
square_peaks = load_peak_data(square_peak_file)
triangle_peaks = load_peak_data(triangle_peak_file)
sine_peaks = load_peak_data(sine_peak_file)
square_waveform = load_waveform_data(square_wave_file)
triangle_waveform = load_waveform_data(triangle_wave_file)
sine_waveform = load_waveform_data(sine_wave_file)

# Check if we have data to plot
data_available = {
    "Sine": sine_waveform is not None,
    "Square": square_waveform is not None,
    "Triangle": triangle_waveform is not None
}

# Filter selected waveforms to only available ones
plots_to_create = [w for w in waveforms_to_plot if data_available.get(w, False)]

if len(plots_to_create) > 0:
    
    # Initialize settings for each plot if not exists
    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {}
    
    for waveform in plots_to_create:
        if waveform not in st.session_state.plot_settings:
            st.session_state.plot_settings[waveform] = {
                'cluster_fraction': 0.9,
                'manual_bar_width': False,
                'bar_width': 3.0,
                'show_grid': True,
                'inversion_mode': 'None',
                'time_threshold': 125.0,
                'invert_current': False,
                'use_abs_intensity': True,
                'show_error_bars': False,
                'xlabel_text': 'Time (μs)',
                'ylabel_left_text': 'Intensity (a.u.)',
                'ylabel_right_text': 'Current (mA)',
                'xlabel_size': 16,
                'ylabel_size': 16,
                'xlabel_weight': 'bold',
                'ylabel_weight': 'bold',
                'tick_label_size': 13,
                'tick_width': 1.5,
                'tick_length': 6,
                'axis_linewidth': 1.5,
                'show_legend': True,
                'legend_fontsize': 11,
                'legend_framealpha': 0.95,
                'legend_x': 0.02,
                'legend_y': 0.98,
                'legend_ncol': 1,
                'show_panel_label': True,
                'panel_label_size': 18,
                'panel_label_x': 0.02,
                'panel_label_y': 0.98,
                'current_linewidth': 2.5,
                'current_time_offset': 0.0,
                'current_y_offset': 0.0,
                'current_scale': 1.0,
                'time_min': 0,
                'time_max': 300,
                'intensity_min': 0.0,
                'intensity_max': 0.3,
                'current_min': -150.0,
                'current_max': 150.0,
                'use_auto_limits': False
            }
    
    # Set matplotlib font
    rcParams['font.family'] = font_family
    
    # Create figure with subplots
    num_plots = len(plots_to_create)
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot each selected waveform
    for idx, waveform_type in enumerate(plots_to_create):
        ax = axes[idx]
        ax2 = ax.twinx()
        
        settings = st.session_state.plot_settings[waveform_type]
        
        # Set axis line width
        for spine in ax.spines.values():
            spine.set_linewidth(settings['axis_linewidth'])
        for spine in ax2.spines.values():
            spine.set_linewidth(settings['axis_linewidth'])
        
        # Get appropriate data
        if waveform_type == "Square":
            peak_data = square_peaks
            wave_data = square_waveform
        elif waveform_type == "Triangle":
            peak_data = triangle_peaks
            wave_data = triangle_waveform
        else:  # Sine
            peak_data = sine_peaks
            wave_data = sine_waveform
        
        # Plot intensity bars if peak data exists with clustered layout
        if peak_data is not None:
            time_col = [col for col in peak_data.columns if 'Time' in col][0]
            
            # Find wavelength columns
            wavelength_cols = []
            col_310 = [col for col in peak_data.columns if '310' in col]
            col_337 = [col for col in peak_data.columns if '337' in col]
            col_696 = [col for col in peak_data.columns if '696' in col]
            
            if col_310:
                wavelength_cols.append(('310.0 nm', col_310[0], color_310))
            if col_337:
                wavelength_cols.append(('337.0 nm', col_337[0], color_337))
            if col_696:
                wavelength_cols.append(('696.0 nm', col_696[0], color_696))
            
            time = peak_data[time_col].values
            
            # Calculate automatic spacing (clustered bar layout)
            unique_times = np.unique(np.sort(time))
            if len(unique_times) >= 2:
                base_spacing = float(np.median(np.diff(unique_times)))
            else:
                base_spacing = 10.0  # default
            
            # Calculate bar positioning
            if settings['manual_bar_width']:
                bar_width = settings['bar_width']
                cluster_width = bar_width * len(wavelength_cols)
            else:
                cluster_width = settings['cluster_fraction'] * base_spacing
                bar_width = cluster_width / max(len(wavelength_cols), 1)
            
            # Determine inversion sign for each time point based on mode
            inversion_sign = np.ones_like(time)  # Default: no inversion
            
            if settings['inversion_mode'] == 'Invert where current < 0':
                # Interpolate current at peak times to determine sign
                if wave_data is not None and 'Time_us' in wave_data.columns:
                    time_wave = wave_data['Time_us'].values
                    current = wave_data['Current'].values
                    # Interpolate current sign at each peak time
                    interp_current = np.interp(time, time_wave, current)
                    inversion_sign = np.where(interp_current < 0, -1.0, 1.0)
            elif settings['inversion_mode'] == 'Invert after time threshold':
                # Invert all bars after threshold time
                inversion_sign = np.where(time >= settings['time_threshold'], -1.0, 1.0)
            
            # Plot each wavelength with proper clustering
            for i, (label, col_name, color) in enumerate(wavelength_cols):
                intensity = peak_data[col_name].values
                
                # Apply absolute values if requested
                if settings['use_abs_intensity']:
                    intensity = np.abs(intensity)
                
                # Apply conditional inversion
                intensity_plot = intensity * inversion_sign
                
                # Calculate position offset for clustering
                offset_start = -cluster_width / 2 + bar_width / 2
                x_pos = time + offset_start + i * bar_width
                
                # Plot bars
                ax.bar(x_pos, intensity_plot, width=bar_width * 0.95, 
                       label=label, color=color, alpha=0.8, edgecolor='none')
                
                # Error bars if enabled
                if settings['show_error_bars']:
                    # Look for error column (with _err suffix or similar)
                    err_col_name = col_name + '_err'
                    if err_col_name in peak_data.columns:
                        err = peak_data[err_col_name].values
                        err_plot = err * inversion_sign
                        ax.errorbar(x_pos, intensity_plot, yerr=np.abs(err_plot), 
                                   fmt='none', ecolor=color, capsize=2, alpha=0.6)
        
        # Plot current waveform with time offset
        if wave_data is not None and 'Time_us' in wave_data.columns:
            time_wave = wave_data['Time_us'].values + settings['current_time_offset']
            current = wave_data['Current'].values
            
            # Apply y scale and offset to current
            current_adjusted = (current * settings['current_scale']) + settings['current_y_offset']
            
            ax2.plot(time_wave, current_adjusted, color=color_current, 
                    linewidth=settings['current_linewidth'], label='Current')
        
        # Axis labels
        ax.set_xlabel(settings['xlabel_text'], fontsize=settings['xlabel_size'], 
                     fontweight=settings['xlabel_weight'])
        
        if peak_data is not None:
            ax.set_ylabel(settings['ylabel_left_text'], fontsize=settings['ylabel_size'], 
                         fontweight=settings['ylabel_weight'])
        else:
            # Hide left y-axis if no peak data
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        
        ax2.set_ylabel(settings['ylabel_right_text'], fontsize=settings['ylabel_size'], 
                      fontweight=settings['ylabel_weight'], color=color_current)
        
        # Tick formatting
        if peak_data is not None:
            ax.tick_params(axis='both', labelsize=settings['tick_label_size'], 
                          width=settings['tick_width'], length=settings['tick_length'])
            
            # If visually inverted, format tick labels to show absolute values
            if settings.get('invert_intensity_visual', False):
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{abs(x):.2f}'))
        else:
            # Only x-axis if no peak data
            ax.tick_params(axis='x', labelsize=settings['tick_label_size'], 
                          width=settings['tick_width'], length=settings['tick_length'])
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        ax2.tick_params(axis='y', labelcolor=color_current, labelsize=settings['tick_label_size'], 
                       width=settings['tick_width'], length=settings['tick_length'])
        
        # Apply axis inversion for current
        if settings['invert_current']:
            ax2.invert_yaxis()
        
        # Set axis limits
        if not settings['use_auto_limits']:
            ax.set_xlim(settings['time_min'], settings['time_max'])
            if peak_data is not None:
                # Ensure intensity limits are non-negative
                int_min = max(0.0, float(settings['intensity_min']))
                int_max = max(int_min + 0.001, float(settings['intensity_max']))
                
                # If visually inverted, flip the signs for actual limits but labels stay positive
                if settings.get('invert_intensity_visual', False):
                    ax.set_ylim(-int_max, -int_min)
                else:
                    ax.set_ylim(int_min, int_max)
            
            if settings['invert_current']:
                ax2.set_ylim(settings['current_max'], settings['current_min'])
            else:
                ax2.set_ylim(settings['current_min'], settings['current_max'])
        else:
            # Auto limits - ensure intensity doesn't go negative (unless visually inverted)
            if peak_data is not None:
                current_ylim = ax.get_ylim()
                if not settings.get('invert_intensity_visual', False):
                    # Normal mode - keep positive
                    if current_ylim[0] < 0 or current_ylim[1] < 0:
                        new_min = max(0.0, min(current_ylim))
                        new_max = max(current_ylim)
                        ax.set_ylim(new_min, new_max)
        
        # Add panel label (no box)
        if settings['show_panel_label']:
            panel_label = chr(97 + idx)  # a, b, c
            ax.text(settings['panel_label_x'], settings['panel_label_y'], f'{panel_label})', 
                   transform=ax.transAxes, 
                   fontsize=settings['panel_label_size'], fontweight='bold', 
                   va='top', ha='left')
        
        # Legend (no box)
        if settings['show_legend'] and peak_data is not None:
            legend = ax.legend(loc='upper left', 
                     bbox_to_anchor=(settings['legend_x'], settings['legend_y']),
                     fontsize=settings['legend_fontsize'], 
                     framealpha=settings['legend_framealpha'], 
                     edgecolor='none',
                     frameon=False,
                     ncol=settings['legend_ncol'])
        
        # Grid
        if settings['show_grid']:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    # Display plot
    st.pyplot(fig)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buf_png = io.BytesIO()
        plt.savefig(buf_png, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf_png.seek(0)
        st.download_button(
            label="📥 Download PNG (300 DPI)",
            data=buf_png,
            file_name="spectral_intensity_plot.png",
            mime="image/png"
        )
    
    with col2:
        buf_pdf = io.BytesIO()
        plt.savefig(buf_pdf, format='pdf', bbox_inches='tight')
        buf_pdf.seek(0)
        st.download_button(
            label="📥 Download PDF",
            data=buf_pdf,
            file_name="spectral_intensity_plot.pdf",
            mime="application/pdf"
        )
    
    with col3:
        buf_svg = io.BytesIO()
        plt.savefig(buf_svg, format='svg', bbox_inches='tight')
        buf_svg.seek(0)
        st.download_button(
            label="📥 Download SVG",
            data=buf_svg,
            file_name="spectral_intensity_plot.svg",
            mime="image/svg+xml"
        )
    
    # Individual plot settings
    st.markdown("---")
    st.header("🎛️ Individual Plot Settings")
    st.info("💡 **Clustered Bar Layout:** Bars automatically positioned within each time step | **Intensity axis:** Values displayed as positive (≥ 0)")
    
    # Create tabs for each plot
    tabs = st.tabs([f"Plot {chr(97 + i)}) - {waveform}" for i, waveform in enumerate(plots_to_create)])
    
    for idx, (tab, waveform) in enumerate(zip(tabs, plots_to_create)):
        with tab:
            settings_key = waveform
            
            # Check if this waveform has peak data
            has_peaks = False
            if waveform == "Square" and square_peaks is not None:
                has_peaks = True
            elif waveform == "Triangle" and triangle_peaks is not None:
                has_peaks = True
            elif waveform == "Sine" and sine_peaks is not None:
                has_peaks = True
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Plot Elements")
                
                if has_peaks:
                    st.markdown("**Bar Clustering:**")
                    st.session_state.plot_settings[settings_key]['cluster_fraction'] = st.slider(
                        "Cluster width fraction", 0.1, 1.0,
                        st.session_state.plot_settings[settings_key]['cluster_fraction'], 0.05,
                        key=f'cluster_{waveform}',
                        help="Fraction of time step filled by bar cluster (0.9 = 90%)"
                    )
                    
                    st.session_state.plot_settings[settings_key]['manual_bar_width'] = st.checkbox(
                        "Manual bar width", 
                        st.session_state.plot_settings[settings_key]['manual_bar_width'],
                        key=f'manual_bar_{waveform}'
                    )
                    
                    if st.session_state.plot_settings[settings_key]['manual_bar_width']:
                        st.session_state.plot_settings[settings_key]['bar_width'] = st.slider(
                            "Bar width (μs)", 0.5, 10.0,
                            st.session_state.plot_settings[settings_key]['bar_width'], 0.1,
                            key=f'bar_w_{waveform}'
                        )
                    
                    st.session_state.plot_settings[settings_key]['use_abs_intensity'] = st.checkbox(
                        "Use absolute intensity values", 
                        st.session_state.plot_settings[settings_key]['use_abs_intensity'],
                        key=f'abs_int_{waveform}',
                        help="Convert negative intensity values to positive"
                    )
                    
                    st.session_state.plot_settings[settings_key]['show_error_bars'] = st.checkbox(
                        "Show error bars", 
                        st.session_state.plot_settings[settings_key]['show_error_bars'],
                        key=f'err_bars_{waveform}',
                        help="Show error bars if _err columns exist"
                    )
                    
                    st.session_state.plot_settings[settings_key]['invert_intensity_visual'] = st.checkbox(
                        "Visually invert intensity bars", 
                        st.session_state.plot_settings[settings_key].get('invert_intensity_visual', False),
                        key=f'inv_int_vis_{waveform}',
                        help="Display bars below zero line while keeping positive axis values"
                    )
                    st.info("✓ Bars auto-clustered within each time step")
                else:
                    st.info("No peak intensity data - showing only current waveform")
                
                st.session_state.plot_settings[settings_key]['show_grid'] = st.checkbox(
                    "Show grid", 
                    st.session_state.plot_settings[settings_key]['show_grid'],
                    key=f'grid_{waveform}'
                )
                
                st.session_state.plot_settings[settings_key]['invert_current'] = st.checkbox(
                    "Invert current axis", 
                    st.session_state.plot_settings[settings_key]['invert_current'],
                    key=f'inv_cur_{waveform}',
                    help="Flip the current axis vertically"
                )
                
                st.markdown("**Current Waveform:**")
                st.session_state.plot_settings[settings_key]['current_linewidth'] = st.slider(
                    "Line width", 0.5, 5.0, 
                    st.session_state.plot_settings[settings_key]['current_linewidth'], 0.1,
                    key=f'lw_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['current_time_offset'] = st.slider(
                    "Time offset (μs)", -200.0, 200.0, 
                    st.session_state.plot_settings[settings_key]['current_time_offset'], 0.5,
                    key=f'time_off_{waveform}',
                    help="Shift current waveform left/right on time axis"
                )
                st.session_state.plot_settings[settings_key]['current_y_offset'] = st.slider(
                    "Y offset (mA)", -100.0, 100.0, 
                    st.session_state.plot_settings[settings_key]['current_y_offset'], 0.5,
                    key=f'y_off_{waveform}',
                    help="Shift current waveform up/down"
                )
                st.session_state.plot_settings[settings_key]['current_scale'] = st.slider(
                    "Y scale factor", 0.1, 3.0, 
                    st.session_state.plot_settings[settings_key]['current_scale'], 0.1,
                    key=f'scale_{waveform}'
                )
            
            with col2:
                st.subheader("📝 Labels & Ticks")
                st.session_state.plot_settings[settings_key]['xlabel_text'] = st.text_input(
                    "X-axis label", 
                    st.session_state.plot_settings[settings_key]['xlabel_text'],
                    key=f'xlab_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['ylabel_left_text'] = st.text_input(
                    "Y-axis left label", 
                    st.session_state.plot_settings[settings_key]['ylabel_left_text'],
                    key=f'ylab_l_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['ylabel_right_text'] = st.text_input(
                    "Y-axis right label", 
                    st.session_state.plot_settings[settings_key]['ylabel_right_text'],
                    key=f'ylab_r_{waveform}'
                )
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.session_state.plot_settings[settings_key]['xlabel_size'] = st.number_input(
                        "X label size", 8, 24, 
                        st.session_state.plot_settings[settings_key]['xlabel_size'],
                        key=f'xlab_s_{waveform}'
                    )
                    st.session_state.plot_settings[settings_key]['ylabel_size'] = st.number_input(
                        "Y label size", 8, 24, 
                        st.session_state.plot_settings[settings_key]['ylabel_size'],
                        key=f'ylab_s_{waveform}'
                    )
                with subcol2:
                    st.session_state.plot_settings[settings_key]['xlabel_weight'] = st.selectbox(
                        "X label weight", ["normal", "bold"],
                        index=["normal", "bold"].index(st.session_state.plot_settings[settings_key]['xlabel_weight']),
                        key=f'xlab_w_{waveform}'
                    )
                    st.session_state.plot_settings[settings_key]['ylabel_weight'] = st.selectbox(
                        "Y label weight", ["normal", "bold"],
                        index=["normal", "bold"].index(st.session_state.plot_settings[settings_key]['ylabel_weight']),
                        key=f'ylab_w_{waveform}'
                    )
                
                st.markdown("**Ticks:**")
                st.session_state.plot_settings[settings_key]['tick_label_size'] = st.slider(
                    "Tick label size", 8, 20, 
                    st.session_state.plot_settings[settings_key]['tick_label_size'],
                    key=f'tick_s_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['tick_width'] = st.slider(
                    "Tick width", 0.5, 3.0, 
                    st.session_state.plot_settings[settings_key]['tick_width'], 0.1,
                    key=f'tick_w_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['tick_length'] = st.slider(
                    "Tick length", 2, 12, 
                    st.session_state.plot_settings[settings_key]['tick_length'],
                    key=f'tick_l_{waveform}'
                )
                st.session_state.plot_settings[settings_key]['axis_linewidth'] = st.slider(
                    "Axis line width", 0.5, 4.0, 
                    st.session_state.plot_settings[settings_key]['axis_linewidth'], 0.1,
                    key=f'axis_w_{waveform}'
                )
            
            with col3:
                st.subheader("📌 Legend & Panel Label")
                
                if has_peaks:
                    st.session_state.plot_settings[settings_key]['show_legend'] = st.checkbox(
                        "Show legend", 
                        st.session_state.plot_settings[settings_key]['show_legend'],
                        key=f'leg_{waveform}'
                    )
                    
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.session_state.plot_settings[settings_key]['legend_x'] = st.slider(
                            "Legend X pos", 0.0, 1.0, 
                            st.session_state.plot_settings[settings_key]['legend_x'], 0.01,
                            key=f'leg_x_{waveform}'
                        )
                        st.session_state.plot_settings[settings_key]['legend_y'] = st.slider(
                            "Legend Y pos", 0.0, 1.0, 
                            st.session_state.plot_settings[settings_key]['legend_y'], 0.01,
                            key=f'leg_y_{waveform}'
                        )
                    with subcol2:
                        st.session_state.plot_settings[settings_key]['legend_fontsize'] = st.slider(
                            "Legend font size", 6, 16, 
                            st.session_state.plot_settings[settings_key]['legend_fontsize'],
                            key=f'leg_fs_{waveform}'
                        )
                        st.session_state.plot_settings[settings_key]['legend_ncol'] = st.slider(
                            "Legend columns", 1, 3, 
                            st.session_state.plot_settings[settings_key]['legend_ncol'],
                            key=f'leg_nc_{waveform}'
                        )
                else:
                    st.info("No peak data - legend not available")
                
                st.markdown("**Panel Label:**")
                st.session_state.plot_settings[settings_key]['show_panel_label'] = st.checkbox(
                    "Show panel label", 
                    st.session_state.plot_settings[settings_key]['show_panel_label'],
                    key=f'pan_{waveform}'
                )
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.session_state.plot_settings[settings_key]['panel_label_x'] = st.slider(
                        "Label X pos", 0.0, 1.0, 
                        st.session_state.plot_settings[settings_key]['panel_label_x'], 0.01,
                        key=f'pan_x_{waveform}'
                    )
                with subcol2:
                    st.session_state.plot_settings[settings_key]['panel_label_y'] = st.slider(
                        "Label Y pos", 0.0, 1.0, 
                        st.session_state.plot_settings[settings_key]['panel_label_y'], 0.01,
                        key=f'pan_y_{waveform}'
                    )
                st.session_state.plot_settings[settings_key]['panel_label_size'] = st.slider(
                    "Label size", 10, 24, 
                    st.session_state.plot_settings[settings_key]['panel_label_size'],
                    key=f'pan_s_{waveform}'
                )
                
                st.markdown("**Axis Ranges:**")
                st.session_state.plot_settings[settings_key]['use_auto_limits'] = st.checkbox(
                    "Auto limits", 
                    st.session_state.plot_settings[settings_key]['use_auto_limits'],
                    key=f'auto_{waveform}'
                )
                
                if not st.session_state.plot_settings[settings_key]['use_auto_limits']:
                    if has_peaks:
                        if st.session_state.plot_settings[settings_key].get('invert_intensity_visual', False):
                            st.info("💡 Visual invert ON: Bars below zero, labels positive | Current: can be negative")
                        else:
                            st.info("💡 Intensity: values ≥ 0 | Current: can be negative")
                    
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.session_state.plot_settings[settings_key]['time_min'] = st.number_input(
                            "Time min (μs)", value=st.session_state.plot_settings[settings_key]['time_min'],
                            key=f'tmin_{waveform}'
                        )
                        if has_peaks:
                            st.session_state.plot_settings[settings_key]['intensity_min'] = st.number_input(
                                "Intensity min", 
                                min_value=0.0,
                                value=float(st.session_state.plot_settings[settings_key]['intensity_min']),
                                format="%.3f", 
                                key=f'imin_{waveform}',
                                help="Minimum intensity (positive, shown as absolute)"
                            )
                        st.session_state.plot_settings[settings_key]['current_min'] = st.number_input(
                            "Current min (mA)", value=st.session_state.plot_settings[settings_key]['current_min'],
                            format="%.1f", key=f'cmin_{waveform}'
                        )
                    with subcol2:
                        st.session_state.plot_settings[settings_key]['time_max'] = st.number_input(
                            "Time max (μs)", value=st.session_state.plot_settings[settings_key]['time_max'],
                            key=f'tmax_{waveform}'
                        )
                        if has_peaks:
                            st.session_state.plot_settings[settings_key]['intensity_max'] = st.number_input(
                                "Intensity max", 
                                min_value=0.001,
                                value=float(st.session_state.plot_settings[settings_key]['intensity_max']),
                                format="%.3f", 
                                key=f'imax_{waveform}',
                                help="Maximum intensity (must be > 0)"
                            )
                        st.session_state.plot_settings[settings_key]['current_max'] = st.number_input(
                            "Current max (mA)", value=st.session_state.plot_settings[settings_key]['current_max'],
                            format="%.1f", key=f'cmax_{waveform}'
                        )
            
            if st.button(f"🔄 Update Plot {chr(97 + idx)})", key=f'update_{waveform}'):
                st.rerun()
    
    # Show data tables
    st.markdown("---")
    st.subheader("📊 Data Tables")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Square Peaks", "Triangle Peaks", "Sine Peaks", "Square Wave", "Triangle Wave", "Sine Wave"]
    )
    
    with tab1:
        if square_peaks is not None:
            st.dataframe(square_peaks, use_container_width=True)
        else:
            st.info("Upload square peak data file")
    
    with tab2:
        if triangle_peaks is not None:
            st.dataframe(triangle_peaks, use_container_width=True)
        else:
            st.info("Upload triangle peak data file")
    
    with tab3:
        if sine_peaks is not None:
            st.dataframe(sine_peaks, use_container_width=True)
        else:
            st.info("Upload sine peak data file (optional)")
    
    with tab4:
        if square_waveform is not None:
            st.dataframe(square_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(square_waveform)} rows")
        else:
            st.info("Upload square waveform data file")
    
    with tab5:
        if triangle_waveform is not None:
            st.dataframe(triangle_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(triangle_waveform)} rows")
        else:
            st.info("Upload triangle waveform data file")
    
    with tab6:
        if sine_waveform is not None:
            st.dataframe(sine_waveform.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(sine_waveform)} rows")
        else:
            st.info("Upload sine waveform data file")

else:
    st.info("👈 Please upload data files from the sidebar to generate visualizations")
    
    with st.expander("📋 File Formats & Features"):
        st.markdown("""
        ### Peak Intensity Files (CSV)
        **Columns:** `Time(us), 310.0 nm, 337.0 nm, 696.00 nm`
        - Optional: Add `_err` suffix columns for error bars (e.g., `310.0 nm_err`)
        
        ### Waveform Files (CSV)
        **Columns:** `Time, Voltage, Current`
        - Time in seconds (auto-converted to μs)
        - Current in mA (can be positive/negative)
        
        ### Key Features:
        **1. Clustered Bar Layout** (from plasma-jet code)
        - Bars automatically grouped within each time step
        - Adjustable cluster width fraction (default 90%)
        - Manual or automatic bar width calculation
        - No overlapping between time groups
        
        **2. Visual Inversion**
        - Display bars below zero line while keeping positive labels
        - Perfect for matching negative current waveform sections
        
        **3. Error Bars**
        - Automatic detection of `_err` columns
        - Toggle on/off per plot
        
        **4. Independent Plot Controls**
        - Each plot has separate settings
        - Time offset for current waveform
        - Precise legend and label positioning
        - Full font and axis customization
        
        **5. Professional Export**
        - PNG (300 DPI), PDF, SVG formats
        - Times New Roman font support
        - Publication-ready quality
        """)

# Footer
st.markdown("---")
st.markdown("**Plasma-Jet Spectral Intensity Dashboard** | Enhanced with clustered bar layout and advanced controls")
