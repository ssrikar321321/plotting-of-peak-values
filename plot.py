import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io

st.set_page_config(page_title="Oscilloscope Spectral Analysis", layout="wide")

st.title("Spectral Intensity & Current Waveform Analysis")
st.markdown("*Visualization of intensity peaks at different wavelengths with oscilloscope current measurements*")

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
                'bar_width': 4.0,
                'show_grid': True,
                'invert_intensity': False,
                'invert_current': False,
                'use_abs_intensity': False,
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
        
        # Plot intensity bars if peak data exists
        if peak_data is not None:
            time_col = [col for col in peak_data.columns if 'Time' in col][0]
            
            # Find wavelength columns
            col_310 = [col for col in peak_data.columns if '310' in col][0]
            col_337 = [col for col in peak_data.columns if '337' in col][0]
            col_696 = [col for col in peak_data.columns if '696' in col][0]
            
            time = peak_data[time_col].values
            int_310 = peak_data[col_310].values
            int_337 = peak_data[col_337].values
            int_696 = peak_data[col_696].values
            
            # Apply absolute values if requested
            if settings['use_abs_intensity']:
                int_310 = np.abs(int_310)
                int_337 = np.abs(int_337)
                int_696 = np.abs(int_696)
            
            # Create bar plots
            width = settings['bar_width']
            ax.bar(time - width, int_310, width=width, 
                   label='310.0 nm', color=color_310, alpha=0.8, edgecolor='none')
            ax.bar(time, int_337, width=width, 
                   label='337.0 nm', color=color_337, alpha=0.8, edgecolor='none')
            ax.bar(time + width, int_696, width=width, 
                   label='696.0 nm', color=color_696, alpha=0.8, edgecolor='none')
        
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
        else:
            # Only x-axis if no peak data
            ax.tick_params(axis='x', labelsize=settings['tick_label_size'], 
                          width=settings['tick_width'], length=settings['tick_length'])
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        ax2.tick_params(axis='y', labelcolor=color_current, labelsize=settings['tick_label_size'], 
                       width=settings['tick_width'], length=settings['tick_length'])
        
        # Apply axis inversions AFTER all plotting
        if settings['invert_intensity'] and peak_data is not None:
            ax.invert_yaxis()
        
        if settings['invert_current']:
            ax2.invert_yaxis()
        
        # Set axis limits (after inversion)
        if not settings['use_auto_limits']:
            ax.set_xlim(settings['time_min'], settings['time_max'])
            if peak_data is not None:
                # Ensure intensity limits are non-negative
                int_min = max(0.0, float(settings['intensity_min']))
                int_max = max(int_min + 0.001, float(settings['intensity_max']))
                
                if settings['invert_intensity']:
                    ax.set_ylim(int_max, int_min)
                else:
                    ax.set_ylim(int_min, int_max)
            
            if settings['invert_current']:
                ax2.set_ylim(settings['current_max'], settings['current_min'])
            else:
                ax2.set_ylim(settings['current_min'], settings['current_max'])
        else:
            # Auto limits - ensure intensity doesn't go negative
            if peak_data is not None:
                current_ylim = ax.get_ylim()
                if current_ylim[0] < 0 or current_ylim[1] < 0:
                    # Adjust to ensure non-negative
                    new_min = max(0.0, min(current_ylim))
                    new_max = max(current_ylim)
                    if settings['invert_intensity']:
                        ax.set_ylim(new_max, new_min)
                    else:
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
    st.info("💡 **Axis Types:** Intensity (left y-axis) must be ≥ 0 | Current (right y-axis) can be positive/negative")
    
    # Create tabs for each plot
    tabs = st.tabs([f"Plot {chr(97 + i)}) - {waveform}" for i, waveform in enumerate(plots_to_create)])
    
    for idx, (tab, waveform) in enumerate(zip(tabs, plots_to_create)):
        with tab:
            settings_key = waveform
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Plot Elements")
                
                # Check if this waveform has peak data
                has_peaks = False
                if waveform == "Square" and square_peaks is not None:
                    has_peaks = True
                elif waveform == "Triangle" and triangle_peaks is not None:
                    has_peaks = True
                elif waveform == "Sine" and sine_peaks is not None:
                    has_peaks = True
                
                if has_peaks:
                    st.session_state.plot_settings[settings_key]['bar_width'] = st.slider(
                        "Bar width (μs)", 1.0, 8.0, 
                        st.session_state.plot_settings[settings_key]['bar_width'], 0.5, 
                        key=f'bar_{waveform}'
                    )
                    st.session_state.plot_settings[settings_key]['use_abs_intensity'] = st.checkbox(
                        "Use absolute intensity values (no negative)", 
                        st.session_state.plot_settings.get(settings_key, {}).get('use_abs_intensity', False),
                        key=f'abs_int_{waveform}',
                        help="Convert all intensity values to positive"
                    )
                else:
                    st.info("No peak intensity data - showing only current waveform")
                
                st.session_state.plot_settings[settings_key]['show_grid'] = st.checkbox(
                    "Show grid", 
                    st.session_state.plot_settings[settings_key]['show_grid'],
                    key=f'grid_{waveform}'
                )
                
                if has_peaks:
                    st.session_state.plot_settings[settings_key]['invert_intensity'] = st.checkbox(
                        "Invert intensity axis", 
                        st.session_state.plot_settings[settings_key]['invert_intensity'],
                        key=f'inv_int_{waveform}',
                        help="Flip the intensity axis vertically (top ↔ bottom)"
                    )
                
                st.session_state.plot_settings[settings_key]['invert_current'] = st.checkbox(
                    "Invert current axis", 
                    st.session_state.plot_settings[settings_key]['invert_current'],
                    key=f'inv_cur_{waveform}',
                    help="Flip the current axis vertically (top ↔ bottom)"
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
                
                # Check if this waveform has peak data
                has_peaks = False
                if waveform == "Square" and square_peaks is not None:
                    has_peaks = True
                elif waveform == "Triangle" and triangle_peaks is not None:
                    has_peaks = True
                elif waveform == "Sine" and sine_peaks is not None:
                    has_peaks = True
                
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
                        st.info("💡 Intensity axis: values must be ≥ 0 | Current axis: can be negative")
                    
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
                                help="Intensity values must be >= 0"
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
                                help="Intensity values must be > 0"
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
    
    with st.expander("📋 Required File Formats"):
        st.markdown("""
        ### Peak Intensity Files (CSV)
        **Columns:** `Time(us), 310.0 nm, 337.0 nm, 696.00 nm`
        - Square: `square_peak_comparison.csv` (Required if showing square)
        - Triangle: `Triangle_peak_comparison.csv` (Required if showing triangle)
        - Sine: Peak data file (Optional - sine can show current waveform only)
        
        **Note:** Intensity values should be non-negative (≥ 0). If your data contains negative values, 
        use the "Use absolute intensity values" option to convert them to positive.
        
        ### Waveform Files (CSV)
        **Columns:** `Time, Voltage, Current`
        - Time in seconds (will be converted to μs)
        - Current in mA (can be positive or negative)
        - Files:
          - `Square Power Calculation_SINGLE CHANNEL 1.csv`
          - `Triangular Power Calculation__SINGLE CHANNEL.csv`
          - `Sine Power Calculation_SINGLE CHANNEL 1.csv`
        
        ### Axis Constraints:
        - **Intensity Axis (left)**: Must be ≥ 0 (peak values cannot be negative)
        - **Current Axis (right)**: Can have both positive and negative values
        - **Time Axis**: Typically starts at 0, measured in microseconds (μs)
        """)

# Footer
st.markdown("---")
st.markdown("**Oscilloscope Spectral Intensity Dashboard** | Data visualization for plasma spectroscopy measurements")
